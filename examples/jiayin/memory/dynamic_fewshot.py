# ok，那我知道了，很简单的检索few shot ， 然后评估器用一个agent就行了吧
# 检索 匹配 query， 返回 (query, res) 作为fewshot
# 评估器agent 输入 q，r 来管理 记忆库，动态删除
# 还有历史删除，这个怎么做到呢，通过 调用次数+评分 也存在记忆库里面，让agent 动态删除/基于规则删除

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent


load_dotenv()

@dataclass
class FewShotRecord:
    """动态 few-shot 条目."""

    id: str
    query: str
    response: str
    score: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    score_sum: float = 0.0
    score_count: int = 0
    last_used: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "FewShotRecord":
        return FewShotRecord(
            id=payload["id"],
            query=payload["query"],
            response=payload["response"],
            score=payload.get("score", 0.0),
            usage_count=payload.get("usage_count", 0),
            success_count=payload.get("success_count", 0),
            failure_count=payload.get("failure_count", 0),
            score_sum=payload.get("score_sum", 0.0),
            score_count=payload.get("score_count", 0),
            last_used=payload.get("last_used", datetime.now(timezone.utc).isoformat()),
            created_at=payload.get("created_at", datetime.now(timezone.utc).isoformat()),
            embedding=list(payload.get("embedding", [])),
            metadata=payload.get("metadata", {}),
        )

    def public_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        data.pop("embedding", None)
        total_feedback = self.success_count + self.failure_count
        average_score = self.score_sum / self.score_count if self.score_count else None
        success_rate = self.success_count / total_feedback if total_feedback else None
        data["average_score"] = average_score
        data["success_rate"] = success_rate
        return data


class DynamicFewShotStore:
    """文件持久化的 dynamic few-shot 存储."""

    def __init__(self, storage_path: Path, embed_model):
        self.storage_path = storage_path
        self.embed_model = embed_model
        self.entries: List[FewShotRecord] = []
        self._ensure_storage_file()
        self._load_entries()

    def _ensure_storage_file(self) -> None:
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def _load_entries(self) -> None:
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.entries = [FewShotRecord.from_dict(item) for item in raw]

    def _persist(self) -> None:
        serialised = [entry.to_dict() for entry in self.entries]
        self.storage_path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")

    def _compute_embedding(self, query: str, response: str) -> List[float]:
        text = f"{query}\n\n{response}".strip()
        vector = self.embed_model.embed_query(text)
        return list(vector)

    def add_entry(
        self,
        query: str,
        response: str,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FewShotRecord:
        initial_score = float(score) if score is not None else 0.0
        entry = FewShotRecord(
            id=str(uuid.uuid4()),
            query=query,
            response=response,
            score=initial_score,
            usage_count=0,
            embedding=self._compute_embedding(query, response),
            metadata=metadata or {},
        )
        if score is not None:
            entry.score_sum = initial_score
            entry.score_count = 1
        self.entries.append(entry)
        self._persist()
        return entry

    def remove_entries(self, entry_ids: List[str]) -> List[str]:
        existing_ids = {entry.id for entry in self.entries}
        to_remove = [entry_id for entry_id in entry_ids if entry_id in existing_ids]
        if not to_remove:
            return []
        self.entries = [entry for entry in self.entries if entry.id not in to_remove]
        self._persist()
        return to_remove

    def update_score(self, entry_id: str, score: float) -> Optional[FewShotRecord]:
        for entry in self.entries:
            if entry.id == entry_id:
                entry.score = score
                entry.last_used = datetime.now(timezone.utc).isoformat()
                self._persist()
                return entry
        return None

    def record_usage(self, entry_id: str, score_delta: Optional[float] = None) -> Optional[FewShotRecord]:
        for entry in self.entries:
            if entry.id == entry_id:
                entry.usage_count += 1
                entry.last_used = datetime.now(timezone.utc).isoformat()
                if score_delta is not None:
                    entry.score += score_delta
                    entry.score_sum += entry.score
                    entry.score_count += 1
                self._persist()
                return entry
        return None

    def register_feedback(
        self,
        entry_id: str,
        *,
        success: Optional[bool] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[FewShotRecord]:
        for entry in self.entries:
            if entry.id == entry_id:
                if success is True:
                    entry.success_count += 1
                elif success is False:
                    entry.failure_count += 1
                if score is not None:
                    entry.score = score
                    entry.score_sum += score
                    entry.score_count += 1
                if metadata:
                    entry.metadata.update(metadata)
                entry.last_used = datetime.now(timezone.utc).isoformat()
                self._persist()
                return entry
        return None

    def bulk_record_usage(self, entry_ids: List[str]) -> None:
        touched = False
        now = datetime.now(timezone.utc).isoformat()
        ids = set(entry_ids)
        for entry in self.entries:
            if entry.id in ids:
                entry.usage_count += 1
                entry.last_used = now
                touched = True
        if touched:
            self._persist()

    def get_snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        ordered = sorted(self.entries, key=lambda item: (item.score, item.usage_count), reverse=True)
        if limit is not None:
            ordered = ordered[:limit]
        return [entry.public_dict() for entry in ordered]

    def prune_by_rules(
        self,
        min_score: Optional[float] = None,
        max_usage: Optional[int] = None,
        min_average_score: Optional[float] = None,
        min_success_rate: Optional[float] = None,
    ) -> List[str]:
        removed: List[str] = []
        for entry in list(self.entries):
            should_remove = False
            if min_score is not None and entry.score < min_score:
                should_remove = True
            if min_average_score is not None:
                avg = entry.score_sum / entry.score_count if entry.score_count else None
                if avg is None or avg < min_average_score:
                    should_remove = True
            if min_success_rate is not None:
                total_feedback = entry.success_count + entry.failure_count
                success_rate = (entry.success_count / total_feedback) if total_feedback else None
                if success_rate is None or success_rate < min_success_rate:
                    should_remove = True
            if max_usage is not None and entry.usage_count >= max_usage:
                should_remove = True
            if should_remove:
                removed.append(entry.id)
                self.entries.remove(entry)
        if removed:
            self._persist()
        return removed

    def _entry_priority(self, entry: FewShotRecord) -> tuple[float, float, float, int, int, float]:
        average_score = entry.score_sum / entry.score_count if entry.score_count else None
        effective_score = average_score if average_score is not None else entry.score
        success_total = entry.success_count + entry.failure_count
        success_rate = (entry.success_count / success_total) if success_total else 0.0
        try:
            last_used_ts = datetime.fromisoformat(entry.last_used).timestamp()
        except Exception:  # pragma: no cover - 容错解析失败
            last_used_ts = 0.0
        return (
            effective_score,
            success_rate,
            entry.score,
            entry.usage_count,
            -entry.failure_count,
            last_used_ts,
        )

    def auto_trim(self, max_entries: int) -> List[str]:
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        if len(self.entries) <= max_entries:
            return []
        ordered = sorted(self.entries, key=self._entry_priority, reverse=True)
        to_remove = ordered[max_entries:]
        removed_ids = [entry.id for entry in to_remove]
        if not removed_ids:
            return []
        self.remove_entries(removed_ids)
        return removed_ids

    def ensure_embeddings(self) -> None:
        updated = False
        for entry in self.entries:
            if not entry.embedding:
                entry.embedding = self._compute_embedding(entry.query, entry.response)
                updated = True
        if updated:
            self._persist()

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.25) -> List[FewShotRecord]:
        if not self.entries:
            return []
        self.ensure_embeddings()
        query_vec = np.array(self.embed_model.embed_query(query))
        query_norm = np.linalg.norm(query_vec) + 1e-8
        scores: List[tuple[FewShotRecord, float]] = []
        for entry in self.entries:
            entry_vec = np.array(entry.embedding)
            denom = (np.linalg.norm(entry_vec) + 1e-8) * query_norm
            similarity = float(entry_vec.dot(query_vec) / denom)
            if similarity >= min_similarity:
                scores.append((entry, similarity))
        if not scores:
            return []
        scores.sort(key=lambda item: item[1], reverse=True)
        top_entries = [item[0] for item in scores[:top_k]]
        self.bulk_record_usage([entry.id for entry in top_entries])
        return top_entries


class DynamicFewShotRetrievalTool(Tool):
    name = "dynamic_few_shot_retriever"
    description = "根据查询返回 dynamic few-shot 片段，结果包含 query 与 response。"
    inputs = {
        "query": {
            "type": "string",
            "description": "工作代理的当前查询。"
        },
        "top_k": {
            "type": "integer",
            "description": "返回的记忆条数，可选。",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, memory_store: DynamicFewShotStore, default_top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.memory_store = memory_store
        self.default_top_k = default_top_k

    def forward(self, query: str, top_k: Optional[int] = None) -> str:
        entries = self.memory_store.retrieve(query, top_k or self.default_top_k)
        payload = [entry.public_dict() for entry in entries]
        return json.dumps({"few_shots": payload}, ensure_ascii=False, indent=2)


class DynamicFewShotManagementTool(Tool):
    name = "dynamic_few_shot_manager"
    description = (
        "根据评估结果选择性地新增或删除 dynamic few-shot 样本，可查看快照。"
        "你的score评分范围是0到10的整数，10表示非常有帮助，0表示完全无帮助。"
        "- 仅在确信当前 query/response 极具参考价值时调用 add，同时补充必要的 score 与 metadata 帮助后续排序。"
        "- 当样本接近 7 条时，你可以选择 remove 你觉得较弱的样本，并提供对应的 entry_ids。"
        "系统会在样本数量超过 7 条时自动移除较弱条目，因此请聚焦评估质量与是否留存。"
    )
    inputs = {
        "action": {"type": "string", "description": "操作类型，仅支持 add/remove/snapshot"},
        "query": {"type": "string", "description": "问题文本", "nullable": True},
        "response": {"type": "string", "description": "答案文本", "nullable": True},
        "entry_ids": {"type": "array", "description": "要删除的条目 id 列表", "nullable": True},
        "score": {"type": "number", "description": "样本的质量评分，可选", "nullable": True},
        "metadata": {"type": "object", "description": "附加元数据", "nullable": True},
        "limit": {"type": "integer", "description": "快照条数", "nullable": True},
    }
    output_type = "string"

    def __init__(self, memory_store: DynamicFewShotStore, **kwargs):
        super().__init__(**kwargs)
        self.memory_store = memory_store

    def forward(
        self,
        action,
        query=None,
        response=None,
        entry_ids=None,
        score=None,
        metadata=None,
        limit=None,
    ) -> str:
        if action not in {"add", "remove", "snapshot"}:
            return json.dumps({"status": "error", "message": f"不支持的 action: {action}"}, ensure_ascii=False)
        if action == "add":
            if not query or not response:
                return json.dumps({"status": "error", "message": "add 操作需要提供 query 与 response"}, ensure_ascii=False)
            entry = self.memory_store.add_entry(
                query=query,
                response=response,
                score=score,
                metadata=metadata,
            )
            return json.dumps({"status": "added", "entry": entry.public_dict()}, ensure_ascii=False)

        if action == "remove":
            removed_ids = self.memory_store.remove_entries(entry_ids or [])
            return json.dumps({"status": "removed", "entry_ids": removed_ids}, ensure_ascii=False)

        if action == "snapshot":
            snapshot = self.memory_store.get_snapshot(limit)
            return json.dumps({"status": "ok", "snapshot": snapshot}, ensure_ascii=False, indent=2)
        return json.dumps({"status": "error", "message": f"未知 action: {action}"}, ensure_ascii=False)


load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 初始化向量化模型与记忆库
embeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
except Exception as exc:  # pragma: no cover - 仅在模型下载失败时触发
    raise RuntimeError(f"无法加载嵌入模型 {EMBEDDING_MODEL_NAME}: {exc}")

DYNAMIC_FEW_SHOT_FILE = Path(__file__).with_name("dynamic_few_shot_store.json")
dynamic_few_shot_store = DynamicFewShotStore(storage_path=DYNAMIC_FEW_SHOT_FILE, embed_model=embeddings)


MAX_FEW_SHOT_ENTRIES = 7


# Tool 实例
# retrieval_tool = DynamicFewShotRetrievalTool(memory_store=dynamic_few_shot_store)
# 不需要agent检索


def fetch_few_shots(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """返回用于 few-shot 的 (query, response)。"""

    entries = dynamic_few_shot_store.retrieve(query, top_k=top_k)
    return [entry.public_dict() for entry in entries]

def create_dynamic_few_shot_agent(memory_store: DynamicFewShotStore, model) -> CodeAgent:
    few_shot_management_tool = DynamicFewShotManagementTool(memory_store=dynamic_few_shot_store)

    DYNAMIC_FEW_SHOT_MANAGER_PROMPT = """
    你负责评估 query-response 对的质量，并决定是否将其添加到 dynamic few-shot 样本库中，或从中删除不合适的样本。
    """

    dynamic_few_shot_agent = CodeAgent(
        tools=[few_shot_management_tool],
        model=model,
        instructions=DYNAMIC_FEW_SHOT_MANAGER_PROMPT.strip(),
        max_steps=2,
        verbosity_level=1,
        # description="Memory management agent that decides how to maintain the few-shot store.",
    )
    return dynamic_few_shot_agent


__all__ = [
    "dynamic_few_shot_store",
    "fetch_few_shots",
    "create_dynamic_few_shot_agent",
]


# copy this：
'''
def sum_positive_numbers_buggy(numbers):
    """
    计算列表中所有正数的和
    """
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num

    return total


# 只需要通过这个case
print(f"sum_odd_numbers_buggy([1, 2, 3, 4, 5, 6])")
'''
if __name__ == "__main__":

    few_shot_management_tool = DynamicFewShotManagementTool(memory_store=dynamic_few_shot_store)

    DYNAMIC_FEW_SHOT_MANAGER_PROMPT = """
    你负责评估 query-response 对的质量，并决定是否将其添加到 dynamic few-shot 样本库中，或从中删除不合适的样本。
    """

    # 评估/管理 Agent
    model = LiteLLMModel(
        model_id="deepseek/deepseek-chat",
        api_key=os.environ.get("deepseek"),
    )

    dynamic_few_shot_agent = CodeAgent(
        tools=[few_shot_management_tool],
        model=model,
        instructions=DYNAMIC_FEW_SHOT_MANAGER_PROMPT.strip(),
        max_steps=2,
        verbosity_level=1,
        # description="Memory management agent that decides how to maintain the few-shot store.",
    )

    code_fix_agent = CodeAgent(
        model=model,
        tools=[],
        instructions="你是一个代码修复专家。",
        max_steps=3,
        verbosity_level=1,
    )
    while True:
        sample_query = input("请输入代码修复问题，连续输入类似的问题可以有帮助（输入 'exit' 退出）：")
        if sample_query == "exit":
            break
        fewshots = fetch_few_shots(sample_query, top_k=2)

        answer = code_fix_agent.run(
            task="修复这个函数中的问题",
            additional_args={
                "few_shots": fewshots,
                "code": sample_query,
                }
            )

        dynamic_few_shot_agent.run(
            task="请评估以下代码修复建议的质量，并根据评估结果维护 dynamic few-shot 样本库",
            additional_args={
                "query": sample_query,
                "response": answer,
                "this_turn_few_shots": fewshots,
            },
        )

        trimmed_ids = dynamic_few_shot_store.auto_trim(MAX_FEW_SHOT_ENTRIES)

        if fewshots:
            print("检索到的 dynamic few-shot 条目:")
            for item in fewshots:
                print(json.dumps(item, ensure_ascii=False, indent=2))
        else:
            print("样本库为空或无匹配项，可通过 dynamic_few_shot_manager 工具新增条目。")

        if trimmed_ids:
            print("自动清理的条目 ID:")
            for entry_id in trimmed_ids:
                print(entry_id)

        snapshot = few_shot_management_tool(
            action="snapshot",
            limit=5,
        )
        print("当前 dynamic few-shot 快照:")
        print(snapshot)
