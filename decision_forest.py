import time
import os, json, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from torch_scatter import scatter_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

import openai
from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai_client = OpenAI(api_key="")


ds_client = OpenAI(api_key="", base_url="https://api.deepseek.com")

nb_client = OpenAI(
    api_key="",
    base_url="https://api.studio.nebius.com/v1/",
)


# Import Dawid-Skene consensus
try:
    from consensus import DawidSkeneModel, list2array
    CONSENSUS_AVAILABLE = True
except ImportError:
    print("[Warning] consensus.py not found. Dawid-Skene voting will not be available.")
    CONSENSUS_AVAILABLE = False



# ================= Basic Utilities =================

def ensure_dir(p: str): 
    os.makedirs(p, exist_ok=True)


def read_csv_strict(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def align_frames(clean: pd.DataFrame, dirty: pd.DataFrame):
    if list(clean.columns) != list(dirty.columns):
        dirty = dirty[clean.columns]
    n = min(len(clean), len(dirty))
    return clean.iloc[:n].reset_index(drop=True), dirty.iloc[:n].reset_index(drop=True)


def compute_ground_truth(clean: pd.DataFrame, dirty: pd.DataFrame) -> pd.DataFrame:
    return (dirty != clean).astype(int)


# ================= CatalogColumn =================

@dataclass
class CatalogColumn:
    name: str
    inferred_type: str
    domain: Optional[List[str]] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    regex_hint: Optional[str] = None


def load_catalog_from_json(json_path: str) -> Dict[str, CatalogColumn]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    catalog = {}
    for col, info in raw.items():
        dtype = info["dataType"]
        if dtype == "numeric":
            itype = "numeric"
            min_val = info.get("statistics", {}).get("min")
            max_val = info.get("statistics", {}).get("max")
        elif dtype == "datetime":
            itype = "date"
            min_val = max_val = None
        elif dtype == "bool":
            itype = "category"
            min_val = max_val = None
        elif dtype == "string":
            if info.get("isCategorical", False):
                itype = "category"
            else:
                itype = "text"
            min_val = max_val = None
        else:
            itype = "text"
            min_val = max_val = None

        domain = info.get("samples", None) if itype == "category" else None
        catalog[col] = CatalogColumn(
            name=col, inferred_type=itype, domain=domain,
            min_val=min_val, max_val=max_val, regex_hint=None
        )
    return catalog


# ================= NodeSpec & Tree =================

@dataclass
class NodeSpec:
    node_id: str
    kind: str   # "rule" | "gnn" | "leaf"
    name: str = ""
    code: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    true_child: Optional[str] = None
    false_child: Optional[str] = None
    leaf_value: Optional[bool] = None



class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr='mean')  # mean aggregator
        self.msg_mlp = nn.Linear(in_dim + edge_dim, out_dim)
        self.update_mlp = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_dim]
        # edge_attr: [num_edges, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # concat neighbor embedding + edge embedding
        m = torch.cat([x_j, edge_attr], dim=-1)
        return F.relu(self.msg_mlp(m))

    def aggregate(self, inputs, index):
        # mean aggregate
        return scatter_mean(inputs, index, dim=0)

    def update(self, aggr_out, x):
        # update node embedding with skip connection
        return F.relu(self.update_mlp(torch.cat([x, aggr_out], dim=-1)))


class EdgeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=32):
        super().__init__()

        self.conv1 = EdgeSAGEConv(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, edge_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, edge_attr, edge_pairs):
        # x: node embeddings
        # edge_attr: explicit edge embeddings
        # edge_pairs: (src, dst) for each evaluated cell (tuple_i, attr_j)

        h = self.conv1(x, edge_index, edge_attr)
        h = self.conv2(h, edge_index, edge_attr)

        src, dst = edge_pairs
        h_edge = torch.cat([h[src], h[dst]], dim=-1)

        return self.fc(h_edge).squeeze(-1)


class GNNWrapper:
    def __init__(self, model, data, row_offset, col_offset):
        self.model = model
        self.data = data
        self.row_offset = row_offset
        self.col_offset = col_offset

    def predict_cell(self, df: pd.DataFrame, row_id: int, col: str):
        """预测单个 cell 是否有 error"""
        self.model.eval()
        col_id = list(df.columns).index(col)
        src = torch.tensor([row_id], dtype=torch.long, device=DEVICE)
        dst = torch.tensor([self.col_offset + col_id], dtype=torch.long, device=DEVICE)
        edge_pair = torch.stack([src, dst], dim=0)
        with torch.no_grad():
            logit = self.model(self.data.x.to(DEVICE), self.data.edge_index.to(DEVICE), self.data.edge_attr.to(DEVICE), edge_pair)
            prob = torch.sigmoid(logit).item()
            return prob, prob > 0.5



# ========= 简单 MLP =========
class MLPNode(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


# ========= MLP Wrapper =========
class MLPWrapper:
    def __init__(self, model, feat_dim=32):
        self.model = model
        self.feat_dim = feat_dim

    def predict_cell(self, df, row_id: int, col: str):
        """预测单个 cell 是否有 error"""
        self.model.eval()
        val = df.iloc[row_id][col]
        feat = hash_embed(val, dim=self.feat_dim)

        x = torch.tensor(feat, dtype=torch.float, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
            return prob, prob > 0.5


# ----------------- Executable Node -----------------

class ExecutableNode:
    def __init__(self, spec: NodeSpec, gnn_model=None, df=None):
        self.spec = spec
        self.gnn_model = gnn_model
        self.df = df
        self.func = None

        if spec.kind == "rule" and spec.code:
            local_env = {}
            try:
                exec(spec.code, {}, local_env)
                self.func = local_env.get("rule_fn", None)
            except Exception as e:
                print(f"[Warning] Failed to compile rule node {spec.node_id}: {e}")

    def route(self, row: pd.Series, col: str) -> bool:
        if self.spec.kind == "rule" and self.func:
            return bool(self.func(row, col))
        elif self.spec.kind == "gnn" and self.gnn_model:
            prob, is_err = self.gnn_model.predict_cell(self.df, row.name, col)
            self.last_prob = prob    
            return is_err         
        return False


class CellDecisionTree:
    def __init__(self, nodes: Dict[str, ExecutableNode], root_id: str, tree_id: int = 0):
        self.nodes = nodes
        self.root_id = root_id
        self.tree_id = tree_id

    def traverse_rowcol(self, row: pd.Series, col: str):
        nid = self.root_id
        path = []
        while True:
            node = self.nodes[nid]
            path.append(nid)
            if node.spec.kind == "leaf":
                return bool(node.spec.leaf_value), path
            go_true = node.route(row, col)
            nid = node.spec.true_child if go_true else node.spec.false_child
            if nid is None:
                return False, path


# ================= LLM Call =================

def call_llm_generate(catalog: Dict[str, CatalogColumn], sample_df: pd.DataFrame, 
                      llm_model="gpt-5", partition_id: int = 0, log_dir: str = None, results_dir: str = None):
    """Generate decision tree for a partition using LLM"""
    cat_desc = {c: asdict(v) for c, v in catalog.items()}
    preview = sample_df.to_dict(orient="records")

    example_output = """
### Example Output Format
```json
{
  "nodes": [
    {
      "node_id": "root",
      "kind": "rule",
      "name": "check_gpa_range",
      "code": "def rule_fn(row, col):\n    if col != 'GPA': return False\n    try:\n        gpa = float(row.get(col, -1))\n    except:\n        return True\n    return gpa < 0.0 or gpa > 4.0",
      "true_child": "leaf_error",
      "false_child": "gnn_student_check"
    },
    {
      "node_id": "gnn_student_check",
      "kind": "gnn",
      "name": "gnn_student_check",
      "true_child": "check_major_validity",
      "false_child": "check_graduation_year"
    },
    {
      "node_id": "check_major_validity",
      "kind": "rule",
      "name": "check_major_validity",
      "code": "def rule_fn(row, col):\n    if col != 'Major': return False\n    valid = {'Computer Science','Mathematics','Biology','History'}\n    return str(row.get(col,'')).strip() not in valid",
      "true_child": "leaf_error",
      "false_child": "leaf_ok"
    },
    {
      "node_id": "check_graduation_year",
      "kind": "rule",
      "name": "check_graduation_year",
      "code": "def rule_fn(row, col):\n    if col != 'GraduationYear': return False\n    try:\n        year = int(row.get(col,0))\n    except:\n        return True\n    return year < 2000 or year > 2030",
      "true_child": "leaf_error",
      "false_child": "leaf_ok"
    },
    {
      "node_id": "leaf_error",
      "kind": "leaf",
      "leaf_value": true
    },
    {
      "node_id": "leaf_ok",
      "kind": "leaf",
      "leaf_value": false
    }
  ],
  "sample_labels": [
    {
      "row_id": 0,
      "column": "GPA",
      "is_error": true,
      "path": ["root","leaf_error"]
    },
    {
      "row_id": 1,
      "column": "Major",
      "is_error": false,
      "path": ["root","gnn_student_check","check_major_validity","leaf_ok"]
    },
    {
      "row_id": 2,
      "column": "GraduationYear",
      "is_error": true,
      "path": ["root","gnn_student_check","check_graduation_year","leaf_error"]
    }
  ]
}
"""

    label_requirements = """
### Labeling Requirements:
- For every sampled row and every column, you must output one entry in `sample_labels`.  
- Each entry must include the **row_id, column name, is_error (bool), and the full path of node_ids from root to leaf**.  
- Flag entries with repeated tokens as errors.
- Generate only confident valiation rules: for categorical columns, restrict to valid categories from the catalog; for non-categorical columns, infer format rules from the samples. Don't generate rules that you are not confident. 
- Detect errors in the data, including empty values, invalid characters (non-English letters except common symbols), suspicious formats (e.g., email-like strings in names), typos or misspellings.
- Sentences must have exactly one space between words and after periods; multiple or missing spaces are errors.
- You must strictly enforce the following error rules:
    Invalid formats: Rate must be a decimal without trailing zeros (invalid formats: "7.0", "07"); ounces format must be 12.0 oz;
    Invalid values: Negative numbers (e.g., ounces = -5) or illegal characters.
    Domain violations: Values outside allowed categories (e.g., gender not in categories).
"""


    prompt = f"""
You are to design a **shallow decision tree** for detecting erroneous CELLS in a table.  
Each node must have **clear responsibilities** and avoid combining all rules in one node.  

### Design Constraints:
1. **Input**: (row, col)  
2. **Node Types**:  
   - "rule": must provide *executable Python code* for a function:
     ```python
     def rule_fn(row, col) -> bool:
        import ...
         # return True if the cell is erroneous, False otherwise
     ```
     Each rule node should handle **only one type of check** (e.g., numeric range, format, or cross-column consistency).
     Imports should be written inside the function, not globally. 
   - "gnn": for **complex relational checks** (e.g., functional dependencies, conditional FDs, denial constraints).  
     The node name must reflect the purpose (e.g., "gnn_fd_check", "gnn_cfd_check").  
   - "leaf": must have a `leaf_value` of `true` or `false`.  

3. The decision tree should be **shallow** (4-8 levels).  
   - Each node should be specialized, not overloaded with many unrelated checks.  
   - There should be at least one gnn node whose children are not leaf nodes. 

{label_requirements}

### Output Format:
Your response must be a single JSON object with two fields:  
- `"nodes"`: list of NodeSpec objects.  
- `"sample_labels"`: list of objects with `row_id`, `column`, `is_error`, and `path`.  

#### Example Output
{example_output}

Task
Generate such a decision tree and labels for the following data (Partition {partition_id}):

Full data catalog:
{json.dumps(cat_desc, ensure_ascii=False)}

Sample rows:
{json.dumps(preview, ensure_ascii=False)}

Important:

Keep rules focused and modular.

Ensure at least one GNN node with a meaningful name and responsibility.

Return ONLY the JSON (no explanations).
"""

    # 记录开始时间
    start_time = time.time()
    token_info = {"prompt": 0, "completion": 0, "total": 0}

    try:
        if llm_model == "cache":
            cache_json = os.path.join(results_dir, f"partition_{partition_id}/llm_output.json")
            if not os.path.exists(cache_json):
                return None, None, None, None
            with open(cache_json, "r", encoding="utf-8") as f:
                js = json.load(f)
            # metrics_json = os.path.join(results_dir, "metrics.json")
            # if not os.path.exists(metrics_json):
            metrics_p_json = os.path.join(results_dir, f"partition_{partition_id}/metrics.json")
            if not os.path.exists(metrics_p_json):
                return None, None, None, None
            with open(metrics_p_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            token_info = data["tokens"]
            duration = data["llm_time"]
        else:  # OpenAI
            # response = nb_client.chat.completions.create(
            # response = ds_client.chat.completions.create(
            response = openai_client.chat.completions.create(
                model=llm_model,
                # messages=[{"role": "system", "content": "You are a precise data quality expert."}, {"role": "user", "content": prompt}],
                temperature=1.0,
                messages=[{"role": "system", "content": "You are a precise data quality expert."}, {"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            print("LLM output: ", text)
            start, end = text.find("{"), text.rfind("}") + 1
            js = json.loads(text[start:end])
            # token 统计
            token_info = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }

            duration = time.time() - start_time

        if log_dir:
            output_log_path = os.path.join(log_dir, f"llm_output.json")
            with open(output_log_path, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)
            ensure_dir(log_dir)
            input_log_path = os.path.join(log_dir, f"llm_input.txt")
            with open(input_log_path, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"LLM INPUT LOG - Partition {partition_id}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Model: {llm_model}\n")
                f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
                f.write("="*80 + "\n")
                f.write("SYSTEM PROMPT:\n")
                f.write("="*80 + "\n")
                f.write("You are a precise data quality expert.\n\n")
                f.write("="*80 + "\n")
                f.write("USER PROMPT:\n")
                f.write("="*80 + "\n")
                f.write(prompt)
                f.write("\n" + "="*80 + "\n")

        nodes_spec = [NodeSpec(**d) for d in js["nodes"]]
        sample_labels = js.get("sample_labels", [])
        print(f"[Partition {partition_id}] LLM call successful (tokens={token_info['total']}, time={duration:.2f}s)")

        return nodes_spec, sample_labels, token_info, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"[Partition {partition_id}] LLM call failed after {duration:.2f}s: {e}")
        raise



def normalize_labels_row_ids(sample_labels: list, sample_indices: list, partition_id: int):
    """
    把 label['row_id'] 统一转为全局行号。
    规则：
      - 如果所有 row_id 都在 [0, len(sample_indices))，视为局部下标 => 映射到全局
      - 否则视为已经是全局行号 => 原样返回
    """
    if not sample_labels:
        return sample_labels

    row_ids = [lab.get("row_id") for lab in sample_labels if "row_id" in lab]
    if not row_ids:
        return sample_labels

    # 判断是否全是局部下标
    is_all_local = all(isinstance(r, int) and 0 <= r < len(sample_indices) for r in row_ids)

    if is_all_local:
        mapped = []
        for lab in sample_labels:
            r = lab["row_id"]
            try:
                lab = dict(lab) 
                lab["row_id"] = int(sample_indices[r])
                mapped.append(lab)
            except Exception:
                raise ValueError(
                    f"[Partition {partition_id}] Invalid local row_id {r} for partition size {len(sample_indices)}"
                )
        return mapped
    else:
        return sample_labels


# ================= GNN Training =================

def hash_embed(val: str, dim=64):
    """把 cell 的值 hash 成固定随机 embedding"""
    h = abs(hash(str(val))) % 108
    np.random.seed(h % 231)
    return np.random.randn(dim)

def build_graph_from_table_for_gnn_node(df: pd.DataFrame, labels: List[dict],
                                        node_id: str, feat_dim=32):
    """
    为指定 GNN 节点构建 bipartite 图：
      - label = 下一个节点的 ID（字符串形式，不做 True/False 推断）
      - 若该节点不在 path 中，则 label = None
    """
    n_rows, n_cols = df.shape
    row_offset, col_offset = 0, n_rows
    num_nodes = n_rows + n_cols
    x = np.zeros((num_nodes, feat_dim), dtype=np.float32)

    # ==== Step 1. 构造 label_dict: (row, col) -> next_node_id ====
    label_dict = {}
    for lab in labels:
        path = lab.get("path", [])
        if node_id in path:
            idx = path.index(node_id)
            if idx + 1 < len(path):
                next_node = path[idx + 1]
                label_dict[(lab["row_id"], lab["column"])] = next_node

    # ==== Step 2. 图结构 ====
    edge_src, edge_dst, edge_feat = [], [], []
    edge_y, edge_label_mask = [], []

    for r in range(n_rows):
        for cidx, c in enumerate(df.columns):
            val = df.iloc[r, cidx]
            feat = hash_embed(val, feat_dim)
            src, dst = r, col_offset + cidx
            edge_src += [src, dst]
            edge_dst += [dst, src]
            edge_feat += [feat, feat]

            # 是否存在 label
            y = label_dict.get((r, c))
            edge_y += [y, y]
            edge_label_mask += [y is not None, y is not None]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        # edge_attr=torch.tensor(edge_feat, dtype=torch.float),
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float)
    )

    data.edge_y = edge_y  # 字符串 label
    data.edge_label_mask = torch.tensor(edge_label_mask, dtype=torch.bool)
    data.row_offset = row_offset
    data.col_offset = col_offset
    return data

def train_gnn_for_node(df: pd.DataFrame, labels: List[dict],
                       node_spec, feat_dim=64, epochs=200, partition_id=0):
    """为单个 GNN 节点训练模型，根据 sample 路径的下一个节点进行监督"""
    data = build_graph_from_table_for_gnn_node(df, labels, node_spec.node_id, feat_dim)

    # ==== GPU 搬运 ====
    data.x = data.x.to(DEVICE)
    data.edge_index = data.edge_index.to(DEVICE)
    data.edge_attr = data.edge_attr.to(DEVICE)

    # 将 next_node 映射为 0/1
    label_num = []
    for y in data.edge_y[::2]:  # 取正向边
        if y is None:
            label_num.append(-1)
        elif y == node_spec.true_child:
            label_num.append(1)
        elif y == node_spec.false_child:
            label_num.append(0)
        else:
            label_num.append(-1)
    label_num = np.array(label_num)
    mask = label_num != -1

    if not mask.any():
        print(f"[Partition {partition_id}] No training data for node {node_spec.name}")
        return None

    model = EdgeGNN(node_dim=feat_dim, edge_dim=feat_dim).to(DEVICE)   # <<< 模型放到 GPU
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    edge_pairs = data.edge_index[:, ::2][:, mask].to(DEVICE)
    y = torch.tensor(label_num[mask], dtype=torch.float, device=DEVICE)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        # logits = model(data.x, data.edge_index, edge_pairs)
        logits = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_pairs=edge_pairs,
        )
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    print(f"[Partition {partition_id}] Node {node_spec.name} trained ({mask.sum()} samples) on {DEVICE}")
    return GNNWrapper(model, data, data.row_offset, data.col_offset)


# ================= Metrics =================

def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


import os
import json
import numpy as np
import pandas as pd
from typing import List
from consensus import list2array, DawidSkeneModel


import numpy as np
import pandas as pd
import os
from typing import List


def dawid_skene_consensus(partition_reports: List[pd.DataFrame], dirty: pd.DataFrame, outdir: str, 
                          results_dir: str = None) -> pd.DataFrame:
    """
    Apply Dawid-Skene EM algorithm for consensus using efficient matrix loading
    
    Args:
        partition_reports: List of prediction dataframes (can be None if results_dir provided)
        dirty: Original dirty dataframe
        outdir: Output directory for saving results
        results_dir: Directory containing partition results with prediction matrices
        
    Returns:
        DataFrame with consensus predictions
    """
    if not CONSENSUS_AVAILABLE:
        print("[Error] Dawid-Skene consensus requires consensus.py")
        raise ImportError("consensus.py not available")
    
    print(f"[Consensus] Applying Dawid-Skene EM algorithm...")
    
    num_rows, num_cols = len(dirty), len(dirty.columns)
    num_workers = len(partition_reports) if partition_reports else 0
    
    # ============= 高效加载：直接从 matrix 文件读取 =============
    if results_dir:
        print(f"[Consensus] Loading predictions from matrix files...")
        prediction_matrices = []
        
        for partition_id in range(num_workers if num_workers > 0 else 1000):  # 自动检测分区数
            partition_dir = os.path.join(results_dir, f"partition_{partition_id}")
            matrix_path = os.path.join(partition_dir, "predictions_matrix.npz")
            
            if not os.path.exists(matrix_path):
                if partition_id == 0:
                    raise FileNotFoundError(f"No prediction matrix found at {matrix_path}")
                break  # 已读取所有分区
            
            # 加载压缩的 numpy 矩阵（非常快）
            loaded = np.load(matrix_path)
            pred_matrix = loaded['predictions']  # shape: (num_rows, num_cols)
            prediction_matrices.append(pred_matrix)
            print(f"[Consensus] Loaded partition {partition_id} matrix: {pred_matrix.shape}")
        
        num_workers = len(prediction_matrices)
        print(f"[Consensus] Loaded {num_workers} prediction matrices")
    else:
        # Fallback: 从 DataFrame 构建（旧方法，较慢）
        print(f"[Consensus] Building matrices from DataFrames (slower)...")
        prediction_matrices = []
        for worker_id, report_df in enumerate(partition_reports):
            pred_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
            for _, row in report_df.iterrows():
                row_idx = int(row["row_id"])
                col_name = row["column"]
                col_idx = dirty.columns.get_loc(col_name)
                pred_matrix[row_idx, col_idx] = int(row["prediction"])
            prediction_matrices.append(pred_matrix)
    
    # ============= 构建 Dawid-Skene 输入张量 =============
    # Shape: (num_tasks, num_workers, 2)
    # 每个 task = (row_id, col_id)，扁平化为 1D
    num_tasks = num_rows * num_cols
    
    print(f"[Consensus] Building dataset tensor: {num_tasks} tasks, {num_workers} workers...")
    
    # 使用 numpy 高效构建（避免 Python 循环）
    dataset_tensor = np.zeros((num_tasks, num_workers, 2), dtype=np.float32)
    
    for worker_id, pred_matrix in enumerate(prediction_matrices):
        # 扁平化预测矩阵
        pred_flat = pred_matrix.flatten()  # shape: (num_tasks,)
        
        # 对于每个 task，设置相应的类别计数
        # pred_flat[i] ∈ {0, 1}
        dataset_tensor[np.arange(num_tasks), worker_id, pred_flat] = 1.0
    
    # ============= 运行 Dawid-Skene =============
    from consensus import DawidSkeneModel
    
    print(f"[Consensus] Running EM with {num_tasks} tasks and {num_workers} workers...")
    model = DawidSkeneModel(class_num=2, max_iter=100, tolerance=1e-4)
    marginal_predict, error_rates, worker_reliability, predict_label = model.run(dataset_tensor)
    
    # ============= 提取结果 =============
    # predict_label: (num_tasks, 2) -> reshape to (num_rows, num_cols, 2)
    predict_label_reshaped = predict_label.reshape(num_rows, num_cols, 2)
    
    # 获取每个 cell 的预测
    consensus_pred_matrix = (predict_label_reshaped[:, :, 1] > predict_label_reshaped[:, :, 0]).astype(np.int8)
    confidence_matrix = np.max(predict_label_reshaped, axis=2)
    
    # ============= 构建输出 DataFrame =============
    consensus_results = []
    for row_id in range(num_rows):
        for col_idx, col_name in enumerate(dirty.columns):
            consensus_results.append({
                "row_id": row_id,
                "column": col_name,
                "consensus_prediction": int(consensus_pred_matrix[row_id, col_idx]),
                "prob_error": float(predict_label_reshaped[row_id, col_idx, 1]),
                "prob_ok": float(predict_label_reshaped[row_id, col_idx, 0]),
                "confidence": float(confidence_matrix[row_id, col_idx])
            })
    
    consensus_df = pd.DataFrame(consensus_results)
    
    # ============= 保存 worker reliability =============
    reliability_data = {
        "worker_id": list(range(num_workers)),
        "reliability": [worker_reliability[i] for i in range(num_workers)]
    }
    reliability_df = pd.DataFrame(reliability_data)
    reliability_df.to_csv(os.path.join(outdir, "worker_reliability.csv"), index=False)
    
    print(f"[Consensus] Worker reliability:")
    for i, rel in enumerate(worker_reliability.values()):
        print(f"  Partition {i}: {rel:.4f}")
    
    # ============= 保存 consensus matrix (可选，用于快速重用) =============
    consensus_matrix_path = os.path.join(outdir, "consensus_predictions_matrix.npz")
    np.savez_compressed(
        consensus_matrix_path,
        predictions=consensus_pred_matrix,
        confidence=confidence_matrix
    )
    print(f"[Consensus] Saved consensus matrix to {consensus_matrix_path}")
    
    return consensus_df


def majority_voting_consensus(partition_reports, dirty, strategy="majority", device="cuda"):
    """
    Perform majority voting across partition reports using GPU (PyTorch).

    Args:
        partition_reports: list of pd.DataFrame, each with ["row_id","column","prediction"]
        dirty: original dirty DataFrame (to get row/col shape)
        device: "cuda" or "cpu"
    Returns:
        pd.DataFrame with consensus predictions
    """

    n_rows, n_cols = dirty.shape
    partitions = len(partition_reports)

    # 列名映射成索引
    col2idx = {col: j for j, col in enumerate(dirty.columns)}

    # 初始化一个 3D tensor: [partitions, n_rows, n_cols]
    preds_tensor = torch.zeros((partitions, n_rows, n_cols), dtype=torch.int32, device=device)

    # 把每个分区的 prediction 填到对应位置
    for p, df in enumerate(partition_reports):
        rows = torch.tensor(df["row_id"].values, dtype=torch.long, device=device)
        cols = torch.tensor([col2idx[c] for c in df["column"]], dtype=torch.long, device=device)
        vals = torch.tensor(df["prediction"].values, dtype=torch.int32, device=device)
        preds_tensor[p, rows, cols] = vals

    # 在 partition 维度上做投票
    vote_sum = preds_tensor.sum(dim=0)           # [n_rows, n_cols]
    vote_count = torch.tensor(partitions, device=device)

    final_pred = (vote_sum > (vote_count / 2)).int()
    agreement = vote_sum.float() / vote_count    # 一致率

    # 转成 DataFrame
    out = []
    for j, col in enumerate(dirty.columns):
        col_df = pd.DataFrame({
            "row_id": torch.arange(n_rows, device=device).cpu().numpy(),
            "column": col,
            "consensus_prediction": final_pred[:, j].cpu().numpy(),
            "vote_sum": vote_sum[:, j].cpu().numpy(),
            "vote_count": int(vote_count.item()),
            "agreement": agreement[:, j].cpu().numpy()
        })
        out.append(col_df)

    return pd.concat(out, ignore_index=True)



def build_tree_for_partition(partition_id: int, sample_indices: List[int], 
                             dirty: pd.DataFrame, clean: pd.DataFrame, 
                             gt_df: pd.DataFrame, catalog: Dict[str, CatalogColumn],
                             llm_model: str, outdir: str, results_dir: str, pred_model: str) -> Tuple[int, CellDecisionTree, List[dict]]:
    """Build a single decision tree for a partition (thread-safe)"""
    try:
        print(f"[Partition {partition_id}] Starting tree construction")
        
        partition_dir = os.path.join(outdir, f"partition_{partition_id}")
        ensure_dir(partition_dir)
        df_sample = dirty.loc[sample_indices].reset_index(drop=True)

        # 调用 LLM
        specs, sample_labels, token_info, duration = call_llm_generate(
            catalog, df_sample, llm_model, partition_id, log_dir=partition_dir, results_dir=results_dir
        )
        if specs is None:
            return None, None, None, None
        sample_labels = normalize_labels_row_ids(sample_labels, sample_indices, partition_id)

        # 训练 GNN
        gnn_time_total = 0
        gnn_wrappers = {}
        for node in specs:
            if node.kind == "gnn":
                node_labels = [lab for lab in sample_labels if node.node_id in lab.get("path", [])]
                start = time.time()
                gnn_wrappers[node.name] = train_gnn_for_node(
                    dirty, node_labels, node, feat_dim=64, epochs=10, partition_id=partition_id
                )
                gnn_time_total += time.time() - start

        tree = CellDecisionTree(
            {s.node_id: ExecutableNode(s, gnn_model=gnn_wrappers.get(s.name), df=dirty) for s in specs},
            specs[0].node_id,
            tree_id=partition_id
        )

        # 保存 tree 规格和 sample_labels
        with open(os.path.join(partition_dir, "tree_spec.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(n) for n in specs], f, indent=2, ensure_ascii=False)
        pd.DataFrame(sample_labels).to_csv(os.path.join(partition_dir, "sample_labels.csv"), index=False)

        # ---------------------- 全表预测 (并行化 n worker) ----------------------
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        def _predict_batch(rows: pd.DataFrame, dirty: pd.DataFrame, clean: pd.DataFrame, gt_df: pd.DataFrame, tree, partition_id: int):
            batch_results = []
            for i, row in rows.iterrows():
                if i % 500 == 0:
                    print(f"[Partition {partition_id}] Processing row {i}/{len(dirty)}")
                for j, col in enumerate(dirty.columns):
                    pred, path = tree.traverse_rowcol(row, col)
                    batch_results.append({
                        "row_id": int(i),
                        "column": col,
                        "dirty_value": dirty.iloc[i, j],
                        "clean_value": clean.iloc[i, j],
                        "prediction": int(pred),
                        "ground_truth": int(gt_df.iloc[i, j]),
                        "path": " -> ".join(path),
                        "path_length": len(path)
                    })
            return batch_results

        partition_predictions = []
        n_workers = 10
        chunk_size = int(np.ceil(len(dirty) / n_workers))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for w in range(n_workers):
                start = w * chunk_size
                end = min((w + 1) * chunk_size, len(dirty))
                if start < end:
                    rows_chunk = dirty.iloc[start:end]
                    futures.append(executor.submit(
                        _predict_batch, rows_chunk, dirty, clean, gt_df, tree, partition_id
                    ))

            for f in as_completed(futures):
                partition_predictions.extend(f.result())

        # 转成 DataFrame
        partition_report_df = pd.DataFrame(partition_predictions)

        # 🔹保证顺序和 dirty 一致
        partition_report_df.sort_values(by=["row_id", "column"], inplace=True)
        partition_report_df.reset_index(drop=True, inplace=True)

        # 保存 CSV report
        partition_report_df.to_csv(os.path.join(partition_dir, "full_table_report.csv"), index=False)

        # ---------------------- 保存 Prediction Matrix ----------------------
        # 使用 numpy 的 memmap 或直接保存为 .npy，支持快速加载
        num_rows, num_cols = len(dirty), len(dirty.columns)
        pred_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)  # 使用 int8 节省空间
        
        for _, row_data in partition_report_df.iterrows():
            row_idx = int(row_data["row_id"])
            col_name = row_data["column"]
            col_idx = dirty.columns.get_loc(col_name)
            pred_matrix[row_idx, col_idx] = int(row_data["prediction"])
        
        # 保存为压缩的 .npz 格式（节省空间）
        np.savez_compressed(
            os.path.join(partition_dir, "predictions_matrix.npz"),
            predictions=pred_matrix
        )
        
        print(f"[Partition {partition_id}] Saved prediction matrix: shape {pred_matrix.shape}")

        # ---------------------- Metrics ----------------------
        partition_metrics = compute_metrics(
            partition_report_df["ground_truth"],
            partition_report_df["prediction"]
        )

        # 🔹 合并 token/时间信息
        partition_metrics.update({
            "tokens": token_info,
            "llm_time": duration,
            "gnn_time": gnn_time_total
        })
        with open(os.path.join(partition_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(partition_metrics, f, indent=2, ensure_ascii=False)

        print(f"[Partition {partition_id}] Tree construction complete - F1: {partition_metrics['f1']:.4f}")
        return partition_id, tree, sample_labels, {
            "partition_id": partition_id,
            "tokens": token_info,
            "llm_time": duration,
            "gnn_time": gnn_time_total,
        }
    except Exception as e:
        print(f"[Partition {partition_id}] Failed: {e}")
        raise

def load_historical_results(results_dir: str, num_partitions: int,
                            dirty: pd.DataFrame = None,
                            clean: pd.DataFrame = None,
                            gt_df: pd.DataFrame = None,
                            catalog: Dict[str, CatalogColumn] = None,
                            llm_model: str = None,
                            outdir: str = None,
                            sample: Optional[pd.DataFrame] = None,
                            pred_model: str = "gnn"):
    """
    Load historical partition results from a previous run.
    If a partition report is missing, rebuild that partition from scratch.
    Also ensures prediction matrix files exist for efficient consensus.
    Returns: List of partition report dataframes, plus metrics consistency
    """
    print(f"[Load] Loading historical results from {results_dir}...")
    partition_reports = []
    
    llm_stats = []
    total_tokens = 0
    total_time = 0.0
    gnn_time_total = 0.0

    for partition_id in range(num_partitions):
        partition_dir = os.path.join(results_dir, f"partition_{partition_id}")
        report_path = os.path.join(partition_dir, "full_table_report.csv")
        metrics_path = os.path.join(partition_dir, "metrics.json")
        matrix_path = os.path.join(partition_dir, "predictions_matrix.npz")
        
        # 判断是否需要重建
        needs_rebuild = False
        
        if not os.path.exists(report_path) or not os.path.exists(metrics_path):
            needs_rebuild = True
            print(f"[Load] Partition {partition_id} missing report/metrics. Rebuilding...")
        elif llm_model == "cache":
            needs_rebuild = True
            print(f"[Load] Partition {partition_id} forced rebuild (cache mode)...")
        else:
            # ✅ Report 和 metrics 都存在
            df_report = pd.read_csv(report_path)
            partition_reports.append(df_report)
            
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            token_info = metrics.get("tokens", {"total": 0})
            duration = metrics.get("llm_time", 0.0)
            gnn_time = metrics.get("gnn_time", 0.0)
            
            stats = {
                "partition_id": partition_id,
                "tokens": token_info,
                "llm_time": duration,
                "gnn_time": gnn_time,
            }
            llm_stats.append(stats)
            total_tokens += token_info.get("total", 0)
            total_time += duration
            gnn_time_total += gnn_time
            
            # 检查是否缺少 matrix 文件，如果缺少则从 report 创建
            if not os.path.exists(matrix_path):
                print(f"[Load] Partition {partition_id} missing matrix file, creating from report...")
                _create_matrix_from_report(df_report, dirty, partition_dir)
            else:
                print(f"[Load] Loaded partition {partition_id}: {len(df_report)} predictions")

        # 如果需要重建
        if needs_rebuild:
            if dirty is None or clean is None or gt_df is None or catalog is None or llm_model is None or outdir is None or sample is None:
                raise ValueError("Missing required inputs for rebuilding partitions")

            if "sample_row_id" not in sample.columns:
                raise ValueError("Sample file must contain 'sample_row_id' column to rebuild partitions")

            all_indices = sample["sample_row_id"].astype(int).tolist()
            batch_size = len(all_indices) // num_partitions
            start, end = partition_id * batch_size, (partition_id + 1) * batch_size
            partition_indices = all_indices[start:end]

            _, _, _, stats = build_tree_for_partition(
                partition_id, partition_indices, dirty, clean, gt_df, catalog, llm_model, outdir, results_dir, pred_model
            )
            if stats is None:
                continue
            if not os.path.exists(report_path):
                raise FileNotFoundError(f"[Error] Partition {partition_id} rebuild failed, {report_path} not found")

            df_report = pd.read_csv(report_path)
            partition_reports.append(df_report)
            print(f"[Load] Rebuilt partition {partition_id}: {len(df_report)} predictions")
            llm_stats.append(stats)
            total_tokens += stats["tokens"]["total"]
            total_time += stats["llm_time"]
            gnn_time_total += stats["gnn_time"]

    print(f"[Load] Successfully prepared {len(partition_reports)} partitions")
    return partition_reports, {
        "llm_stats": llm_stats, 
        "total_tokens": total_tokens, 
        "total_time": total_time, 
        "gnn_time_total": gnn_time_total}


def _create_matrix_from_report(report_df: pd.DataFrame, dirty: pd.DataFrame, partition_dir: str):
    """从已有的 report CSV 创建 prediction matrix 文件"""
    import numpy as np
    
    num_rows, num_cols = len(dirty), len(dirty.columns)
    pred_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
    
    for _, row_data in report_df.iterrows():
        row_idx = int(row_data["row_id"])
        col_name = row_data["column"]
        col_idx = dirty.columns.get_loc(col_name)
        pred_matrix[row_idx, col_idx] = int(row_data["prediction"])
    
    matrix_path = os.path.join(partition_dir, "predictions_matrix.npz")
    np.savez_compressed(matrix_path, predictions=pred_matrix)
    print(f"[Load] Created prediction matrix at {matrix_path}")



# ================= Pipeline =================


def run_pipeline(clean_path, dirty_path, sample_path, outdir, llm_model, catalog_json,
                 batch_size=10, max_workers=4, voting="majority", 
                 load_from=None, num_partitions=None, pred_model="gnn"):
    """
    Main pipeline with decision forest approach
    
    Args:
        batch_size: Number of samples per partition
        max_workers: Number of parallel threads for LLM calls
        voting: Voting strategy ("majority", "unanimous", "dawid_skene")
        load_from: Path to load historical results (skip LLM calls)
        num_partitions: Number of partitions to load (required if load_from is set)
    """
    import numpy as np
    
    print(f"[Init] Ensuring output directory: {outdir}")
    ensure_dir(outdir)

    print("[Init] Loading clean and dirty datasets...")
    clean, dirty = align_frames(read_csv_strict(clean_path), read_csv_strict(dirty_path))
    print(f"[Init] Clean shape={clean.shape}, Dirty shape={dirty.shape}")

    print("[Init] Computing ground truth and loading catalog...")
    gt_df = compute_ground_truth(clean, dirty)
    catalog = load_catalog_from_json(catalog_json)
    print(f"[Init] Catalog loaded with {len(catalog)} attributes")

    trees = []
    all_sample_labels = []
    llm_stats = []   # 保存每个 partition 的 token & time
    total_tokens = 0
    total_time = 0.0
    gnn_time_total = 0.0

    # ------------------------
    # Case 1: 从历史结果加载
    # ------------------------
    if load_from:
        print(f"[Load] Loading historical results from {load_from} ...")
        if num_partitions is None:
            raise ValueError("num_partitions must be specified when load_from is used")
        
        partition_reports, partition_stats = load_historical_results(
            load_from, num_partitions,
            dirty=dirty, clean=clean, gt_df=gt_df, catalog=catalog,
            llm_model=llm_model, outdir=outdir,
            sample=pd.read_csv(sample_path, dtype=str, keep_default_na=False) if sample_path else None,
            pred_model=pred_model
        )
        llm_stats = partition_stats["llm_stats"]
        total_tokens = partition_stats["total_tokens"]
        total_time = partition_stats["total_time"]
        gnn_time_total = partition_stats["gnn_time_total"]

        print(f"[Load] Historical results loaded | partitions={num_partitions}, tokens={total_tokens}, time={total_time:.2f}s")

    else:
        # ------------------------
        # Case 2: 从头开始构建树
        # ------------------------
        print(f"[Build] Loading sample file: {sample_path}")
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        sample_idx = sample["sample_row_id"].astype(int).tolist() if "sample_row_id" in sample.columns else list(range(len(sample)))
        print(f"[Build] Loaded {len(sample)} samples, using {len(sample_idx)} indices")

        # 按 batch_size 划分 sample
        partitions = [sample_idx[i:i+batch_size] for i in range(0, len(sample_idx), batch_size)]

        print(f"\n{'='*60}")
        print(f"Decision Forest Configuration")
        print(f"{'='*60}")
        print(f"Total samples: {len(sample_idx)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of partitions: {len(partitions)}")
        print(f"Max parallel workers: {max_workers}")
        print(f"Voting strategy: {voting}")
        print(f"{'='*60}\n")

        partition_reports = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for partition_id, partition_indices in enumerate(partitions):
                print(f"[Dispatch] Partition {partition_id} dispatched with {len(partition_indices)} rows...")
                futures.append(executor.submit(
                    build_tree_for_partition,
                    partition_id, partition_indices, dirty, clean, gt_df, catalog, llm_model, outdir, load_from, pred_model
                ))

            for future in as_completed(futures):
                partition_id, tree, labels, stats = future.result()
                trees.append(tree)
                all_sample_labels.extend(labels)

                partition_dir = os.path.join(outdir, f"partition_{partition_id}")
                report_path = os.path.join(partition_dir, "full_table_report.csv")
                if os.path.exists(report_path):
                    df_report = pd.read_csv(report_path)
                    partition_reports.append(df_report)
                    print(f"[Collect] Partition {partition_id} report loaded: {report_path}")

                # 收集 LLM 调用信息
                llm_stats.append(stats)
                total_tokens += stats["tokens"]["total"]
                total_time += stats["llm_time"]
                gnn_time_total += stats["gnn_time"]

                print(f"[Collect] Partition {partition_id} done | Tokens={stats['tokens']['total']} | "
                      f"LLM time={stats['llm_time']:.2f}s | GNN time={stats['gnn_time']:.2f}s")

        print(f"\n[Build] All {len(trees)} trees built successfully")

    print(f"[Summary] Total tokens used: {total_tokens}")
    print(f"[Summary] Total LLM time: {total_time:.2f}s")
    print(f"[Summary] Total GNN time: {gnn_time_total:.2f}s")

    # ------------------------
    # Consensus voting
    # ------------------------
    print(f"\n[Consensus] Performing ensemble consensus with '{voting}' strategy...")
    start_cons = time.time()
    if voting == "dawid_skene":
        consensus_df = dawid_skene_consensus(partition_reports, dirty, outdir, 
                                            results_dir=load_from if load_from else outdir)
    else:
        consensus_df = majority_voting_consensus(partition_reports, dirty, voting)
    consensus_time = time.time() - start_cons
    print(f"[Consensus] Finished in {consensus_time:.2f}s | Results shape={consensus_df.shape}")

    # ------------------------
    # 🚀 优化：使用 matrix 直接计算 metrics
    # ------------------------
    print("[Metrics] Computing metrics from consensus matrix...")
    metrics_start = time.time()
    
    num_rows, num_cols = len(dirty), len(dirty.columns)
    
    # 加载 consensus prediction matrix
    if voting == "dawid_skene":
        consensus_matrix_path = os.path.join(outdir, "consensus_predictions_matrix.npz")
        if os.path.exists(consensus_matrix_path):
            loaded = np.load(consensus_matrix_path)
            consensus_pred_matrix = loaded['predictions']
            confidence_matrix = loaded['confidence']
            print(f"[Metrics] Loaded consensus matrix from {consensus_matrix_path}")
        else:
            # Fallback: 从 DataFrame 构建
            print(f"[Metrics] Building consensus matrix from DataFrame...")
            consensus_pred_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
            confidence_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)
            for _, row in consensus_df.iterrows():
                r_idx = int(row["row_id"])
                c_name = row["column"]
                c_idx = dirty.columns.get_loc(c_name)
                consensus_pred_matrix[r_idx, c_idx] = int(row["consensus_prediction"])
                confidence_matrix[r_idx, c_idx] = float(row["confidence"])
    else:
        # 从 consensus_df 构建 matrix（majority voting）
        print(f"[Metrics] Building consensus matrix from DataFrame...")
        consensus_pred_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
        vote_sum_matrix = np.zeros((num_rows, num_cols), dtype=np.int32)
        vote_count_matrix = np.zeros((num_rows, num_cols), dtype=np.int32)
        
        for _, row in consensus_df.iterrows():
            r_idx = int(row["row_id"])
            c_name = row["column"]
            c_idx = dirty.columns.get_loc(c_name)
            consensus_pred_matrix[r_idx, c_idx] = int(row["consensus_prediction"])
            vote_sum_matrix[r_idx, c_idx] = int(row["vote_sum"])
            vote_count_matrix[r_idx, c_idx] = int(row["vote_count"])
    
    # Ground truth matrix (已有)
    gt_matrix = gt_df.values.astype(np.int8)
    
    # 保存 predictions.csv (使用 pandas 直接从 matrix 生成)
    preds_df = pd.DataFrame(consensus_pred_matrix, columns=dirty.columns)
    preds_df.to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    print(f"[Output] Predictions saved to {outdir}/predictions.csv")
    
    # 计算 overall metrics（向量化）
    overall = compute_metrics_from_matrices(gt_matrix.flatten(), consensus_pred_matrix.flatten())
    
    # 计算 per-column metrics（向量化）
    per_col = {}
    for col_idx, col_name in enumerate(dirty.columns):
        per_col[col_name] = compute_metrics_from_matrices(
            gt_matrix[:, col_idx], 
            consensus_pred_matrix[:, col_idx]
        )
    
    # 🚀 构建详细的 ensemble_voting_details.csv（优化版）
    print("[Metrics] Generating voting details CSV...")
    
    # 预先构建所有数据（避免逐行循环）
    row_indices = np.repeat(np.arange(num_rows), num_cols)
    col_indices = np.tile(np.arange(num_cols), num_rows)
    
    ensemble_details = {
        "row_id": row_indices,
        "column": [dirty.columns[c] for c in col_indices],
        "dirty_value": dirty.values.flatten(),
        "clean_value": clean.values.flatten(),
        "ground_truth": gt_matrix.flatten(),
        "ensemble_prediction": consensus_pred_matrix.flatten()
    }
    
    if voting == "dawid_skene":
        # 从 consensus_df 提取概率信息
        prob_error_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)
        prob_ok_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)
        
        for _, row in consensus_df.iterrows():
            r_idx = int(row["row_id"])
            c_name = row["column"]
            c_idx = dirty.columns.get_loc(c_name)
            prob_error_matrix[r_idx, c_idx] = float(row["prob_error"])
            prob_ok_matrix[r_idx, c_idx] = float(row["prob_ok"])
        
        ensemble_details["prob_error"] = prob_error_matrix.flatten()
        ensemble_details["prob_ok"] = prob_ok_matrix.flatten()
        ensemble_details["confidence"] = confidence_matrix.flatten()
    else:
        ensemble_details["vote_sum"] = vote_sum_matrix.flatten()
        ensemble_details["vote_count"] = vote_count_matrix.flatten()
        agreement = vote_sum_matrix / np.maximum(vote_count_matrix, 1)
        ensemble_details["agreement"] = agreement.flatten()
    
    ensemble_df = pd.DataFrame(ensemble_details)
    ensemble_df.to_csv(os.path.join(outdir, "ensemble_voting_details.csv"), index=False)
    print(f"[Output] Voting details saved to {outdir}/ensemble_voting_details.csv")
    
    metrics_time = time.time() - metrics_start
    print(f"[Metrics] Metrics computation finished in {metrics_time:.2f}s")

    # ------------------------
    # 保存 metrics.json (加入 LLM 统计)
    # ------------------------
    total_prompt = sum(p["tokens"]["prompt"] for p in llm_stats)
    total_completion = sum(p["tokens"]["completion"] for p in llm_stats)

    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall,
            "per_column": per_col,
            "llm_usage": {
                "per_partition": llm_stats,
                "total_tokens": total_tokens,
                "total_prompt": total_prompt,
                "total_completion": total_completion,
                "total_time_sec": total_time
            },
            "gnn_usage": {
                "total_time_sec": gnn_time_total
            },
            "consensus_usage": {
                "strategy": voting,
                "time_sec": consensus_time
            },
            "metrics_computation_time_sec": metrics_time
        }, f, indent=2, ensure_ascii=False)
    print(f"[Output] Metrics saved to {metrics_path}")

    # ------------------------
    # 打印 summary
    # ------------------------
    print(f"\n{'='*60}")
    print("Decision Forest Results")
    print(f"{'='*60}")
    print(f"Overall Metrics:")
    print(f"  Accuracy:  {overall['accuracy']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1 Score:  {overall['f1']:.4f}")
    print(f"\nLLM Usage Summary:")
    print(f"  Total tokens used: {total_tokens}")
    print(f"  Total LLM time: {total_time:.2f} seconds")
    print(f"  Total GNN time: {gnn_time_total:.2f} seconds")
    print(f"  Metrics computation time: {metrics_time:.2f} seconds")
    print(f"{'='*60}\n")


def compute_metrics_from_matrices(y_true, y_pred):
    """
    高效计算 metrics（使用 numpy 向量化）
    
    Args:
        y_true: Ground truth array (flattened or per-column)
        y_pred: Prediction array (flattened or per-column)
    
    Returns:
        dict: Metrics dictionary
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

# ================= CLI =================

def parse_args():
    ap = argparse.ArgumentParser(description="Decision Forest for Error Detection")
    ap.add_argument("--clean", required=True, help="Path to clean CSV file")
    ap.add_argument("--dirty", required=True, help="Path to dirty CSV file")
    ap.add_argument("--sample", help="Path to sample CSV file (not required if --load_from is used)")
    ap.add_argument("--catalog", required=True, help="Path to catalog JSON file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--llm_model", default="openai/gpt-5", help="LLM model to use")
    ap.add_argument("--batch_size", type=int, default=10, help="Samples per partition")
    ap.add_argument("--max_workers", type=int, default=1, help="Max parallel threads")
    ap.add_argument("--voting", default="dawid_skene", 
                    choices=["majority", "unanimous", "dawid_skene"],
                    help="Voting strategy for ensemble")
    ap.add_argument("--load_from", type=str, default=None,
                    help="Load historical results from this directory (skip LLM calls)")
    ap.add_argument("--num_partitions", type=int, default=None,
                    help="Number of partitions to load (required with --load_from)")
    ap.add_argument("--pred_model", type=str, default="gnn",
                    help="model for prediction gnn/mlp")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments
    if args.load_from and args.num_partitions is None:
        raise ValueError("--num_partitions must be specified when using --load_from")
    
    if not args.load_from and not args.sample:
        raise ValueError("--sample is required when not using --load_from")
    
    run_pipeline(
        clean_path=args.clean,
        dirty_path=args.dirty,
        sample_path=args.sample,
        outdir=args.outdir,
        llm_model=args.llm_model,
        catalog_json=args.catalog,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        voting=args.voting,
        load_from=args.load_from,
        num_partitions=args.num_partitions,
        pred_model=args.pred_model
    )

# Example usage:
# python3 decision_forest.py --clean ./data/hospital_clean.csv --dirty ./data/hospital_error-01.csv --catalog ./data/hospital_error-01_catalog_1.json --outdir ./data/hospital_output
