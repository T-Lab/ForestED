"""
Consensus module for Dawid-Skene algorithm with GPU optimization
"""
import logging
import numpy as np
import torch


def list2array(class_num: int, dataset_list: list) -> np.ndarray:
    """
    Convert a list of annotations into a 3D tensor.
    Shape: (task_num, worker_num, class_num)
    """
    task_num, worker_num = len(dataset_list), len(dataset_list[0])
    dataset_tensor = np.zeros((task_num, worker_num, class_num), dtype=np.float32)
    for task_i in range(task_num):
        for worker_j in range(worker_num):
            for predict_label_k in dataset_list[task_i][worker_j]:
                dataset_tensor[task_i, worker_j, predict_label_k] += 1.0
    return dataset_tensor


class DawidSkeneModel:
    def __init__(
        self,
        class_num: int,
        max_iter: int = 100,
        tolerance: float = 0.01,
        device: str | None = None,
        eps: float = 1e-12,
        batch_size: int = 50000,  # 新增：用于批处理大规模数据
    ) -> None:
        """
        Args:
            class_num: number of classes
            max_iter: max EM iterations
            tolerance: L1 tolerance for convergence on priors and error rates
            device: 'cuda' / 'cpu' / None (auto)
            eps: numerical epsilon to avoid log(0) and division by zero
            batch_size: batch size for processing large datasets (避免 OOM)
        """
        self.class_num = int(class_num)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.eps = float(eps)
        self.batch_size = int(batch_size)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[DawidSkene] Using device: {self.device}, batch_size: {self.batch_size}")

    def run(self, dataset: np.ndarray):
        """
        Args:
            dataset: numpy array of shape (task, worker, class_pred) with counts
        
        Returns:
            marginal_predict: numpy array (class,)
            error_rates: numpy array (worker, class_true, class_pred)
            worker_reliability: dict[int, float]
            predict_label: numpy array (task, class) posterior per task
        """
        T, W, K = dataset.shape
        assert K == self.class_num, "class_num must match dataset last dimension"
        self.task_num, self.worker_num = T, W
        
        print(f"[DawidSkene] Processing {T} tasks, {W} workers, {K} classes")
        
        # 对于大规模数据，使用批处理策略
        use_batching = T > self.batch_size
        
        if use_batching:
            print(f"[DawidSkene] Large dataset detected, using batched processing")
            return self._run_batched(dataset)
        else:
            # 小数据集：直接全部加载到 GPU
            dataset_t = self._to_tensor(dataset)
            self.dataset_tensor = dataset_t
            return self._run_full(dataset_t)
    
    def _run_full(self, dataset_t: torch.Tensor):
        """Full dataset processing (for smaller datasets)"""
        T, W, K = dataset_t.shape
        
        # Initialize predict_label
        predict_label = dataset_t.sum(dim=1)  # (T, K)
        predict_label = self._normalize_lastdim(predict_label)
        
        flag = True
        prev_error_rates = None
        prev_predict_label = None
        iter_num = 0
        
        while flag:
            error_rates = self._m_step(predict_label)
            next_predict_label, log_L = self._e_step_and_ll(predict_label, error_rates)
            
            if iter_num == 0:
                logging.info("%d\t%f", iter_num, float(log_L))
            else:
                marginal_predict = predict_label.mean(dim=0)
                prev_marginal_predict = prev_predict_label.mean(dim=0)
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))
                
                if self._check_condition(float(marginals_diff), float(error_rates_diff), iter_num):
                    flag = False
            
            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1
            
            if iter_num > self.max_iter:
                break
        
        # Final outputs
        marginal_predict = predict_label.mean(dim=0)
        diag = torch.diagonal(prev_error_rates, dim1=1, dim2=2)
        reliability_vec = (diag * marginal_predict.unsqueeze(0)).sum(dim=1)
        worker_reliability = {int(i): float(reliability_vec[i]) for i in range(W)}
        
        return (
            marginal_predict.detach().cpu().numpy(),
            prev_error_rates.detach().cpu().numpy(),
            worker_reliability,
            predict_label.detach().cpu().numpy(),
        )
    
    def _run_batched(self, dataset: np.ndarray):
        """Batched processing for large datasets (避免 OOM)"""
        T, W, K = dataset.shape
        
        # Initialize predict_label on CPU first
        predict_label = np.sum(dataset, axis=1)  # (T, K)
        predict_label = predict_label / (predict_label.sum(axis=1, keepdims=True) + self.eps)
        
        flag = True
        prev_error_rates = None
        prev_marginal = None
        iter_num = 0
        
        while flag:
            # M-step: 批处理累积
            error_rates = self._m_step_batched(dataset, predict_label)
            
            # E-step: 批处理更新
            next_predict_label, log_L = self._e_step_batched(dataset, predict_label, error_rates)
            
            # Check convergence
            if iter_num == 0:
                print(f"[DawidSkene] Iter {iter_num}: log_L = {log_L:.6f}")
            else:
                marginal_predict = next_predict_label.mean(axis=0)
                marginals_diff = np.sum(np.abs(marginal_predict - prev_marginal))
                error_rates_diff = np.sum(np.abs(error_rates - prev_error_rates))
                
                print(f"[DawidSkene] Iter {iter_num}: log_L = {log_L:.6f}, "
                      f"marginals_diff = {marginals_diff:.6e}, error_rates_diff = {error_rates_diff:.6e}")
                
                if marginals_diff < self.tolerance and error_rates_diff < self.tolerance:
                    flag = False
            
            prev_error_rates = error_rates
            prev_marginal = next_predict_label.mean(axis=0)
            predict_label = next_predict_label
            iter_num += 1
            
            if iter_num > self.max_iter:
                break
        
        # Compute worker reliability
        marginal_predict = predict_label.mean(axis=0)
        diag = np.diagonal(error_rates, axis1=1, axis2=2)  # (W, K)
        reliability_vec = (diag * marginal_predict[None, :]).sum(axis=1)
        worker_reliability = {int(i): float(reliability_vec[i]) for i in range(W)}
        
        return marginal_predict, error_rates, worker_reliability, predict_label
    
    def _m_step_batched(self, dataset: np.ndarray, predict_label: np.ndarray) -> np.ndarray:
        """M-step with batching"""
        T, W, K = dataset.shape
        worker_error_rate = np.zeros((W, K, K), dtype=np.float32)
        
        num_batches = int(np.ceil(T / self.batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, T)
            
            # Load batch to GPU
            batch_data = torch.from_numpy(dataset[start_idx:end_idx]).to(self.device)
            batch_pred = torch.from_numpy(predict_label[start_idx:end_idx]).to(self.device)
            
            # Accumulate: einsum over batch
            batch_err = torch.einsum("ti,twk->wik", batch_pred, batch_data)
            worker_error_rate += batch_err.cpu().numpy()
            
            # Free GPU memory
            del batch_data, batch_pred, batch_err
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Normalize
        denom = worker_error_rate.sum(axis=2, keepdims=True).clip(min=self.eps)
        error_rates = worker_error_rate / denom
        return error_rates.clip(min=self.eps)
    
    def _e_step_batched(self, dataset: np.ndarray, predict_label: np.ndarray, 
                        error_rates: np.ndarray) -> tuple:
        """E-step with batching"""
        T, W, K = dataset.shape
        
        # Compute prior
        prior = predict_label.mean(axis=0).clip(min=self.eps)
        log_prior = np.log(prior)
        
        # Precompute log_pi on GPU
        log_pi = torch.from_numpy(np.log(error_rates.clip(min=self.eps))).to(self.device)
        
        next_predict_label = np.zeros((T, K), dtype=np.float32)
        total_log_L = 0.0
        
        num_batches = int(np.ceil(T / self.batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, T)
            
            batch_data = torch.from_numpy(dataset[start_idx:end_idx]).to(self.device)
            
            # Log-likelihood: einsum
            log_likelihood_tl = torch.einsum("twk,wlk->tl", batch_data, log_pi)
            
            # Posterior (log space)
            log_post_unnorm = torch.from_numpy(log_prior).to(self.device).unsqueeze(0) + log_likelihood_tl
            log_norm = torch.logsumexp(log_post_unnorm, dim=1, keepdim=True)
            batch_pred = torch.exp(log_post_unnorm - log_norm)
            
            next_predict_label[start_idx:end_idx] = batch_pred.cpu().numpy()
            total_log_L += log_norm.sum().item()
            
            # Free memory
            del batch_data, log_likelihood_tl, log_post_unnorm, log_norm, batch_pred
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return next_predict_label, total_log_L
    
    def _m_step(self, predict_label: torch.Tensor) -> torch.Tensor:
        """M-step for full GPU processing"""
        worker_error_rate = torch.einsum("ti,twk->wik", predict_label, self.dataset_tensor)
        denom = worker_error_rate.sum(dim=2, keepdim=True).clamp_min(self.eps)
        error_rates = worker_error_rate / denom
        return error_rates.clamp_min(self.eps)
    
    def _e_step_and_ll(self, predict_label: torch.Tensor, error_rates: torch.Tensor) -> tuple:
        """E-step for full GPU processing"""
        prior = predict_label.mean(dim=0).clamp_min(self.eps)
        log_prior = torch.log(prior)
        log_pi = torch.log(error_rates.clamp_min(self.eps))
        
        log_likelihood_tl = torch.einsum("twk,wlk->tl", self.dataset_tensor, log_pi)
        log_post_unnorm = log_prior.unsqueeze(0) + log_likelihood_tl
        log_norm = torch.logsumexp(log_post_unnorm, dim=1, keepdim=True)
        next_predict_label = torch.exp(log_post_unnorm - log_norm)
        log_L = log_norm.sum()
        
        return next_predict_label, float(log_L.detach().item())
    
    def _normalize_lastdim(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return x / s
    
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            t = arr
        else:
            t = torch.from_numpy(arr)
        return t.to(dtype=torch.float32, device=self.device)
    
    def _check_condition(self, marginals_diff: float, error_rates_diff: float, iter_num: int) -> bool:
        return (marginals_diff < self.tolerance and error_rates_diff < self.tolerance)
