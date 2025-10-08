"""
Simplified Branching Agent: 用list返回值实现分支传播

核心思路：
1. Wrapper检测sub-agent分支，返回list of results（而不是单个result）
2. Step-level执行检测action_output是否为list
3. 如果是list，创建多个parent copies，每个继承对应的sub-agent结果
4. 每个parent copy继续执行剩余步骤

无需exception，无需复杂检测
"""

import ast
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import anyio
import anyio.to_thread


# Add smolagents to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import os
import time

from dotenv import load_dotenv

from smolagents import CodeAgent, LiteLLMModel
from smolagents.local_python_executor import fix_final_answer_code
from smolagents.memory import ActionStep, AgentMemory, MemoryStep, TaskStep
from smolagents.memory import ActionStep as SmolagentsActionStep
from smolagents.monitoring import Timing
from smolagents.utils import parse_code_blobs


load_dotenv()

def get_model():
    return LiteLLMModel(
        model_id="deepseek/deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("deepseek"),
        api_base="https://api.deepseek.com",
    )


# ============================================================================
# Phase 1: 数据结构定义
# ============================================================================

@dataclass
class CodeAgentTreeNode:
    """
    树节点 = 单个 MemoryStep + 树结构元数据
    每个节点只存储这一步的 step，完整的 trajectory 通过从根到叶子的路径重建
    """
    # 只存这一步的 step！
    step: MemoryStep

    # Agent 名称
    agent_name: str = "main"

    # 树结构元数据（最小化）
    depth: int = 0
    parent: Optional['CodeAgentTreeNode'] = None
    children: List['CodeAgentTreeNode'] = field(default_factory=list)

    # Sub-agent 的子树
    # key = step_index, value = sub-agent 的 CodeAgentTreeNode
    sub_trees: Dict[int, 'CodeAgentTreeNode'] = field(default_factory=dict)

    def add_child(self, child: 'CodeAgentTreeNode'):
        """添加子分支"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def add_sub_tree(self, step_index: int, sub_tree: 'CodeAgentTreeNode'):
        """为某个 step 添加 sub-agent 的树"""
        self.sub_trees[step_index] = sub_tree

    def get_trajectory(self) -> List[MemoryStep]:
        """从根到当前节点的完整路径（所有 steps）"""
        steps = []
        current = self
        while current:
            steps.insert(0, current.step)
            current = current.parent
        return steps

    def is_leaf(self) -> bool:
        """是否是叶子节点"""
        return len(self.children) == 0

    def to_dict(self) -> Dict:
        """序列化（用于保存/训练）"""
        return {
            'agent_name': self.agent_name,
            'depth': self.depth,
            'step': self.step.dict(),
            'children': [child.to_dict() for child in self.children],
            'sub_trees': {
                step_idx: sub_tree.to_dict()
                for step_idx, sub_tree in self.sub_trees.items()
            }
        }


class CodeAgentTreeMemory:
    """
    树结构管理器 - 只记录根节点
    """

    def __init__(self):
        self.root: Optional[CodeAgentTreeNode] = None

    def set_root(self, step: MemoryStep) -> CodeAgentTreeNode:
        """设置根节点（通常是 SystemPromptStep 或 TaskStep）"""
        self.root = CodeAgentTreeNode(step=step, depth=0)
        return self.root

    def get_all_leaf_nodes(self) -> List[CodeAgentTreeNode]:
        """获取所有叶子节点"""
        if not self.root:
            return []

        leaves = []
        def traverse(node: CodeAgentTreeNode):
            if node.is_leaf():
                leaves.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return leaves

    def get_all_trajectories(self) -> List[List[MemoryStep]]:
        """获取所有叶子节点的完整轨迹（从根到叶子的所有 steps）"""
        return [leaf.get_trajectory() for leaf in self.get_all_leaf_nodes()]

    def to_dict(self) -> Dict:
        """序列化整棵树"""
        return self.root.to_dict() if self.root else {}

    def save(self, filepath: str):
        """保存树到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Tree memory saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CodeAgentTreeMemory':
        """从文件加载树"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 重建树（简化版，只加载结构）
        tree_memory = cls()
        # TODO: 完整重建需要反序列化 AgentMemory
        print(f"Tree memory loaded from: {filepath}")
        return tree_memory

    def visualize(self) -> str:
        """可视化树结构（调试用）"""
        if not self.root:
            return "Empty tree"

        lines = []
        def traverse(node: CodeAgentTreeNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            step_type = type(node.step).__name__

            # 提取关键信息
            extra_info = ""
            if isinstance(node.step, TaskStep):
                task_preview = node.step.task[:40] + "..." if len(node.step.task) > 40 else node.step.task
                extra_info = f': "{task_preview}"'
            elif isinstance(node.step, ActionStep):
                if node.step.is_final_answer:
                    answer_str = str(node.step.action_output)
                    answer_preview = answer_str[:30] + "..." if len(answer_str) > 30 else answer_str
                    extra_info = f' [FINAL: {answer_preview}]'
                elif node.step.error:
                    extra_info = f' [ERROR]'
                elif "Calling sub-agent" in (node.step.observations or ""):
                    extra_info = f' [SUB-AGENT PENDING]'

            line = f"{prefix}{connector}Depth {node.depth} [{node.agent_name}] {step_type}{extra_info}"
            lines.append(line)

            # Sub-trees（显示 sub-agent 的树）
            if node.sub_trees:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                for step_idx, sub_root in node.sub_trees.items():
                    lines.append(f"{sub_prefix}↳ Sub-agent tree:")
                    traverse(sub_root, sub_prefix + "  ", True)

            # Children
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                traverse(child, child_prefix, is_last_child)

        lines.append("CodeAgent Tree:")
        traverse(self.root)
        return "\n".join(lines)


@dataclass
class PendingSubAgentStep:
    """
    表示需要调用 sub-agent 的暂停状态
    """
    code: str  # 生成的代码
    agent_calls: List[Dict]  # 需要调用的 sub-agents
    memory_step: Any = None  # 当前 MemoryStep

    def __repr__(self):
        return f"PendingSubAgentStep(agents={[call['name'] for call in self.agent_calls]})"


@dataclass
class BranchConfig:
    """分支配置"""
    mode: Literal['step_level', 'agent_level', 'none'] = 'none'
    n_branches: int = 2
    branch_at_steps: Optional[int] = None  # None = 所有step都分支, 否则每隔n步分支一次



class BranchingCodeAgent(CodeAgent):
    """支持自动分支的CodeAgent（简化版）"""

    def __init__(
        self,
        tools: list,
        model,
        branch_mode: Literal['step_level', 'agent_level', 'none'] = 'none',
        n_branches: int = 2,
        branch_at_steps: Optional[List[int]] = None,
        branch_config: Optional[BranchConfig] = None,
        **kwargs
    ):
        # 从 kwargs 中移除 branch_config 相关参数（如果存在）
        kwargs.pop('branch_config', None)

        super().__init__(tools=tools, model=model, **kwargs)

        # 支持两种方式设置 branch_config
        if branch_config is not None:
            self.branch_config = branch_config
        else:
            self.branch_config = BranchConfig(
                mode=branch_mode,
                n_branches=n_branches,
                branch_at_steps=branch_at_steps
            )

        self._branch_depth = 0

        # 保存原始 managed_agents（不包装）
        if self.managed_agents:
            self._original_managed_agents = {
                name: agent for name, agent in self.managed_agents.items()
            }

    def copy_for_branch(self):
        """创建一个用于分支的 agent 副本，只 copy memory，共享其他资源"""
        # 创建新 agent（共享 model、tools 等）
        new_agent = BranchingCodeAgent(
            tools=list(self.tools.values()) if self.tools else [],
            model=self.model,
            branch_config=self.branch_config,
            managed_agents=list(self.managed_agents.values()) if self.managed_agents else [],
            additional_authorized_imports=self.additional_authorized_imports,
            executor_type=self.executor_type if hasattr(self, 'executor_type') else 'local',
            executor_kwargs=self.executor_kwargs if hasattr(self, 'executor_kwargs') else None,
            verbosity_level=self.verbosity_level if hasattr(self, 'verbosity_level') else 0,
            max_steps=self.max_steps if hasattr(self, 'max_steps') else 10,
        )

        # Copy memory（深拷贝 steps）
        new_agent.memory.steps = copy.deepcopy(self.memory.steps)
        new_agent.memory.system_prompt = self.memory.system_prompt

        # Copy step_number 和 task
        if hasattr(self, 'step_number'):
            new_agent.step_number = self.step_number
        if hasattr(self, 'task'):
            new_agent.task = self.task
        if hasattr(self, 'name'):
            new_agent.name = self.name

        return new_agent

    # ============================================================================
    # Phase 2: Agent 接口实现
    # ============================================================================

    def _extract_agent_calls(self, code: str) -> List[Dict]:
        """
        解析代码，找出所有 managed_agents 的调用

        Returns:
            List[Dict]: 每个 dict 包含：
                - name: agent 名称
                - args: 调用参数（尽可能提取）
                - line: 行号
                - node: AST node
        """
        if not self.managed_agents:
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        agent_calls = []

        class AgentCallVisitor(ast.NodeVisitor):
            def __init__(self, managed_agent_names):
                self.managed_agent_names = managed_agent_names
                self.calls = []

            def visit_Call(self, node):
                # 检查是否是直接调用 managed_agent
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.managed_agent_names:
                        # 提取参数
                        args_dict = {}

                        # 位置参数：第一个是 task
                        if node.args:
                            # 简单情况：如果是字符串字面量
                            if isinstance(node.args[0], ast.Constant):
                                args_dict['task'] = node.args[0].value
                            else:
                                # 复杂表达式，记录为字符串
                                args_dict['task'] = ast.unparse(node.args[0])

                        # 关键字参数
                        for keyword in node.keywords:
                            if keyword.arg == 'task':
                                if isinstance(keyword.value, ast.Constant):
                                    args_dict['task'] = keyword.value.value
                                else:
                                    args_dict['task'] = ast.unparse(keyword.value)

                        self.calls.append({
                            'name': func_name,
                            'args': args_dict,
                            'line': node.lineno,
                            'node': node
                        })

                self.generic_visit(node)

        visitor = AgentCallVisitor(set(self.managed_agents.keys()))
        visitor.visit(tree)
        agent_calls = visitor.calls

        return agent_calls

    def step_async(self, task: Optional[str] = None) -> Union[ActionStep, PendingSubAgentStep]:
        """
        执行一步，返回：
        - ActionStep: 正常执行完成
        - PendingSubAgentStep: 遇到 sub-agent 调用，需要外层处理
        """

        # 0. 确保 python_executor 有 tools（重要！）
        if hasattr(self, 'python_executor') and self.python_executor:
            self.python_executor.send_tools({**self.tools, **self.managed_agents})
            # Debug: 检查 authorized imports
            # if hasattr(self, 'authorized_imports'):
            #     print(f"[DEBUG] Authorized imports: {self.authorized_imports}")

        # 1. 如果有 task，先添加 TaskStep 到 memory
        if task:
            task_step = TaskStep(task=task)
            self.task = task
            self.memory.steps.append(task_step)
            # 初始化 step_number
            if not hasattr(self, 'step_number') or self.step_number == 0:
                self.step_number = 1

        # 2. 创建 ActionStep 来执行这一步
        memory_step = SmolagentsActionStep(
            step_number=self.step_number,
            timing=Timing(start_time=time.time())
        )

        # 3. 生成代码（调用 LLM）
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        try:
            # 生成模型输出
            chat_message = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
            )
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            memory_step.model_output = output_text
        except Exception as e:
            raise Exception(f"Error generating model output: {e}")

        # 3. 解析代码
        try:
            code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as parse_error:
            # 如果解析失败，可能是 LLM 没有生成代码，记录错误并返回
            # 创建一个简单的错误对象（不需要 logger）
            class SimpleError:
                def __init__(self, message):
                    self.message = message
                def dict(self):
                    return {"type": "ParseError", "message": str(self.message)}
                def __str__(self):
                    return str(self.message)

            memory_step.error = SimpleError(f"Code parsing failed: {parse_error}")
            memory_step.timing.end_time = time.time()
            memory_step.is_final_answer = False  # 不是 final answer，需要重试
            self.memory.steps.append(memory_step)
            self.step_number += 1
            return memory_step

        # 4. 检测 sub-agent 调用
        agent_calls = self._extract_agent_calls(code_action)

        if agent_calls:
            print(f"[DEBUG] Detected agent call in code:")
            print(f"[DEBUG] Code: {code_action[:200]}...")
            print(f"[DEBUG] Agent calls: {agent_calls}")
            # 保存 memory_step（标记为等待 sub-agent）
            memory_step.code_action = code_action
            memory_step.observations = "Calling sub-agent..."
            memory_step.action_output = None
            memory_step.is_final_answer = False
            memory_step.timing.end_time = time.time()
            self.memory.steps.append(memory_step)
            self.step_number += 1
            return memory_step

        # 5. 正常执行代码
        try:
            code_output = self.python_executor(code_action)
            execution_logs = code_output.logs
            output = code_output.output
            is_final = code_output.is_final_answer
        except Exception as exec_error:
            # 代码执行失败，记录错误并返回
            # 创建一个简单的错误对象（不需要 logger）
            class SimpleError:
                def __init__(self, message):
                    self.message = message
                def dict(self):
                    return {"type": "ExecutionError", "message": str(self.message)}
                def __str__(self):
                    return str(self.message)

            memory_step.error = SimpleError(f"Code execution failed: {exec_error}")
            memory_step.timing.end_time = time.time()
            memory_step.is_final_answer = False
            self.memory.steps.append(memory_step)
            self.step_number += 1
            return memory_step

        # 6. 创建 ActionStep
        observation = "Execution logs:\n" + execution_logs
        observation += "\nLast output from code snippet:\n" + str(output)

        memory_step.observations = observation
        memory_step.action_output = output
        memory_step.is_final_answer = is_final
        memory_step.timing.end_time = time.time()

        self.memory.steps.append(memory_step)

        # 递增 step_number（模仿 CodeAgent 的行为）
        self.step_number += 1

        return memory_step

    def continue_step(
        self,
        pending_step: PendingSubAgentStep,
        sub_result: Any
    ) -> ActionStep:
        """
        用 sub-agent 的结果继续执行之前暂停的 step
        """
        code = pending_step.code
        agent_name = pending_step.agent_calls[0]['name']

        # Mock sub-agent
        original_agent = self.managed_agents[agent_name]

        def mocked_agent(task, **kwargs):
            return sub_result

        self.managed_agents[agent_name] = mocked_agent
        self.python_executor.send_tools({**self.tools, **self.managed_agents})

        # 执行代码
        output, logs, is_final = self.python_executor(code)

        # 恢复
        self.managed_agents[agent_name] = original_agent
        self.python_executor.send_tools({**self.tools, **self.managed_agents})

        # 创建 ActionStep
        action_step = ActionStep(
            step_number=self.step_number,
            code_action=code,
            action_output=output,
            observations=logs,
        )
        self.memory.steps.append(action_step)
        self.step_number += 1

        return action_step


# ============================================================================
# Phase 3: TreeRollout 执行器
# ============================================================================

class TreeRollout:
    """
    树状展开执行器
    - 根据 branch_config 递归执行
    - 每个节点只存新增的一个 step
    - 支持 sub-agent 递归
    """

    def __init__(self, root_agent: BranchingCodeAgent):
        self.root_agent = root_agent
        self.tree_memory = CodeAgentTreeMemory()

    async def rollout(self, task: str) -> CodeAgentTreeMemory:
        """
        异步执行任务，返回完整的 TreeMemory

        核心逻辑：
        1. 创建 TaskStep 作为根节点（不执行）
        2. 根据 branch_config 递归执行
        3. 每个节点只存新增的那个 step
        """
        # 创建 TaskStep 作为根节点
        task_step = TaskStep(task=task)
        self.root_agent.task = task
        self.root_agent.memory.steps.append(task_step)

        # 创建根节点
        root_node = self.tree_memory.set_root(task_step)
        root_node.agent_name = self.root_agent.name if hasattr(self.root_agent, 'name') else 'main'

        # 根据 mode 递归执行（从 step 1 开始）
        if self.root_agent.branch_config.mode == 'agent_level':
            await self._execute_agent_level(self.root_agent, root_node, step_num=1)
        elif self.root_agent.branch_config.mode == 'step_level':
            await self._execute_step_level(self.root_agent, root_node, step_num=1)
        else:
            await self._execute_no_branch(self.root_agent, root_node)

        return self.tree_memory

    async def _check_and_force_final_answer(self, agent: BranchingCodeAgent, parent_node: CodeAgentTreeNode) -> bool:
        """
        检查是否达到 max_steps，如果达到则强制生成 final answer

        Returns:
            bool: 如果强制生成了 final answer 返回 True，否则返回 False
        """
        # 使用 agent.step_number 而不是计数 memory 中的 ActionStep
        current_step_num = agent.step_number if hasattr(agent, 'step_number') else 0
        max_steps = agent.max_steps if hasattr(agent, 'max_steps') else 10

        if current_step_num > max_steps:
            # 达到 max_steps，强制生成 final answer
            final_answer = await anyio.to_thread.run_sync(agent.provide_final_answer, agent.task)
            final_step = ActionStep(
                step_number=current_step_num,
                timing=Timing(start_time=time.time(), end_time=time.time()),
                action_output=final_answer.content if hasattr(final_answer, 'content') else final_answer,
                is_final_answer=True,
            )
            agent.memory.steps.append(final_step)
            agent.step_number += 1  # 递增 step_number
            new_node = CodeAgentTreeNode(
                step=final_step,
                agent_name=agent.name if hasattr(agent, 'name') else 'main',
                depth=parent_node.depth + 1
            )
            parent_node.add_child(new_node)
            return True
        return False

    async def _execute_agent_level(self, agent: BranchingCodeAgent, parent_node: CodeAgentTreeNode, step_num: int):
        """
        Agent-level 分支递归执行
        先分支，每个分支独立执行一步，然后递归
        """
        n_branches = agent.branch_config.n_branches

        # Agent-level: 先分支
        branch_agents = [agent.copy_for_branch() for _ in range(n_branches)]
        print(f"[DEBUG] Created {len(branch_agents)} branch agents,{agent.name}")
        # 每个分支并行执行一步
        async def execute_branch(branch_agent):
            # 检查是否达到 max_steps
            if await self._check_and_force_final_answer(branch_agent, parent_node):
                return

            # 执行一步
            prev_len = len(branch_agent.memory.steps)
            step_result = await anyio.to_thread.run_sync(branch_agent.step_async, None)

            # 提取新增的 step
            new_len = len(branch_agent.memory.steps)
            if new_len <= prev_len:
                return

            new_step = branch_agent.memory.steps[-1]
            new_node = CodeAgentTreeNode(
                step=new_step,
                agent_name=branch_agent.name if hasattr(branch_agent, 'name') else 'main',
                depth=parent_node.depth + 1
            )
            parent_node.add_child(new_node)

            # 检查是否有 sub-agent 调用
            sub_agent_result = await self._check_sub_agent_call(branch_agent, new_node)

            if sub_agent_result:
                sub_results, sub_agent_name = sub_agent_result
                # 有 sub-agent 结果，为每个结果创建分支并并行执行
                async def execute_sub_result_branch(sub_result):
                    new_branch = branch_agent.copy_for_branch()
                    # 更新最后一步的 observation 和 action_output
                    # 重要：明确告诉 LLM 结果已返回，不需要再次调用
                    new_branch.memory.steps[-1].observations = (
                        f"Execution logs:\n"
                        f"Sub-agent '{sub_agent_name}' completed successfully.\n"
                        f"\nLast output from code snippet:\n{sub_result}"
                    )
                    new_branch.memory.steps[-1].action_output = sub_result
                    new_branch.memory.steps[-1].is_final_answer = False  # 不是最终答案，需要继续处理
                    # 继续线性执行（agent-level 不再分支）
                    await self._execute_no_branch(new_branch, new_node)

                async with anyio.create_task_group() as tg:
                    for sub_result in sub_results:
                        tg.start_soon(execute_sub_result_branch, sub_result)
            else:
                # 没有 sub-agent，正常继续
                if not (isinstance(step_result, ActionStep) and step_result.is_final_answer):
                    await self._execute_no_branch(branch_agent, new_node)

        async with anyio.create_task_group() as tg:
            for branch_agent in branch_agents:
                tg.start_soon(execute_branch, branch_agent)

    async def _execute_step_level(self, agent: BranchingCodeAgent, parent_node: CodeAgentTreeNode, step_num: int):
        """
        Step-level 分支递归执行
        先判断是否分支，如果分支则先创建分支再并行执行
        """
        n_branches = agent.branch_config.n_branches
        branch_at_steps = agent.branch_config.branch_at_steps

        # 检查是否需要在这个 step 分支
        should_branch = (not branch_at_steps) or (step_num % branch_at_steps == 0)

        if should_branch:
            # 先分支
            branch_agents = [agent.copy_for_branch() for _ in range(n_branches)]

            async def execute_one_branch(branch_agent):
                # 检查是否达到 max_steps
                if await self._check_and_force_final_answer(branch_agent, parent_node):
                    return

                # 执行一步
                prev_len = len(branch_agent.memory.steps)
                step_result = await anyio.to_thread.run_sync(branch_agent.step_async, None)

                if len(branch_agent.memory.steps) <= prev_len:
                    return

                new_step = branch_agent.memory.steps[-1]
                print(f"[DEBUG] execute_branch: Creating new node at depth {parent_node.depth + 1}")
                new_node = CodeAgentTreeNode(
                    step=new_step,
                    agent_name=branch_agent.name if hasattr(branch_agent, 'name') else 'main',
                    depth=parent_node.depth + 1
                )
                parent_node.add_child(new_node)

                # 检查 sub-agent
                sub_agent_result = await self._check_sub_agent_call(branch_agent, new_node)

                if sub_agent_result:
                    sub_results, sub_agent_name = sub_agent_result
                    # 有 sub-agent 结果，为每个结果创建分支并并行执行
                    async def execute_sub_result_branch(sub_result):
                        new_branch = branch_agent.copy_for_branch()
                        new_branch.memory.steps[-1].observations = (
                            f"Execution logs:\n"
                            f"Sub-agent '{sub_agent_name}' completed successfully.\n"
                            f"\nLast output from code snippet:\n{sub_result}"
                        )
                        new_branch.memory.steps[-1].action_output = sub_result
                        new_branch.memory.steps[-1].is_final_answer = False
                        # 继续 step-level 执行
                        await self._execute_step_level(new_branch, new_node, step_num + 1)

                    async with anyio.create_task_group() as tg:
                        for sub_result in sub_results:
                            tg.start_soon(execute_sub_result_branch, sub_result)
                else:
                    # 没有 sub-agent，检查是否完成
                    if not (isinstance(step_result, ActionStep) and step_result.is_final_answer):
                        await self._execute_step_level(branch_agent, new_node, step_num + 1)

            # 并行执行所有分支
            async with anyio.create_task_group() as tg:
                for branch_agent in branch_agents:
                    tg.start_soon(execute_one_branch, branch_agent)
        else:
            # 不分支，直接执行一步
            # 检查是否达到 max_steps
            if await self._check_and_force_final_answer(agent, parent_node):
                return

            prev_len = len(agent.memory.steps)
            step_result = await anyio.to_thread.run_sync(agent.step_async, None)

            if len(agent.memory.steps) <= prev_len:
                return

            new_step = agent.memory.steps[-1]
            new_node = CodeAgentTreeNode(
                step=new_step,
                agent_name=agent.name if hasattr(agent, 'name') else 'main',
                depth=parent_node.depth + 1
            )
            parent_node.add_child(new_node)

            # 检查 sub-agent
            sub_agent_result = await self._check_sub_agent_call(agent, new_node)

            if sub_agent_result:
                sub_results, sub_agent_name = sub_agent_result
                # 并行执行所有 sub-agent 结果分支
                async def execute_sub_result_branch(sub_result):
                    new_branch = agent.copy_for_branch()
                    new_branch.memory.steps[-1].observations = (
                        f"Execution logs:\n"
                        f"Sub-agent '{sub_agent_name}' completed successfully.\n"
                        f"\nLast output from code snippet:\n{sub_result}"
                    )
                    new_branch.memory.steps[-1].action_output = sub_result
                    new_branch.memory.steps[-1].is_final_answer = False
                    await self._execute_step_level(new_branch, new_node, step_num + 1)

                async with anyio.create_task_group() as tg:
                    for sub_result in sub_results:
                        tg.start_soon(execute_sub_result_branch, sub_result)
            else:
                if not (isinstance(step_result, ActionStep) and step_result.is_final_answer):
                    await self._execute_step_level(agent, new_node, step_num + 1)

    async def _execute_no_branch(self, agent: BranchingCodeAgent, parent_node: CodeAgentTreeNode):
        """
        不分支递归执行
        """
        # 检查是否达到 max_steps
        if await self._check_and_force_final_answer(agent, parent_node):
            return

        # 执行一步
        prev_len = len(agent.memory.steps)
        step_result = await anyio.to_thread.run_sync(agent.step_async, None)

        # 提取新增的 step
        if len(agent.memory.steps) <= prev_len:
            return

        new_step = agent.memory.steps[-1]
        new_node = CodeAgentTreeNode(
            step=new_step,
            agent_name=agent.name if hasattr(agent, 'name') else 'main',
            depth=parent_node.depth + 1
        )
        parent_node.add_child(new_node)

        # 检查是否有 sub-agent 调用
        sub_agent_result = await self._check_sub_agent_call(agent, new_node)

        if sub_agent_result:
            sub_results, sub_agent_name = sub_agent_result
            # 有 sub-agent 结果，为每个结果创建分支并并行执行
            async def execute_sub_result_branch(sub_result):
                new_branch = agent.copy_for_branch()
                new_branch.memory.steps[-1].observations = (
                    f"Execution logs:\n"
                    f"Sub-agent '{sub_agent_name}' completed successfully.\n"
                    f"\nLast output from code snippet:\n{sub_result}"
                )
                new_branch.memory.steps[-1].action_output = sub_result
                new_branch.memory.steps[-1].is_final_answer = False
                await self._execute_no_branch(new_branch, new_node)

            async with anyio.create_task_group() as tg:
                for sub_result in sub_results:
                    tg.start_soon(execute_sub_result_branch, sub_result)
        else:
            # 检查是否完成
            if isinstance(step_result, ActionStep) and step_result.is_final_answer:
                return

            # 继续递归
            await self._execute_no_branch(agent, new_node)

    async def _check_sub_agent_call(self, agent: BranchingCodeAgent, node: CodeAgentTreeNode):
        """
        检查当前 step 是否调用了 sub-agent
        如果有，递归执行 sub-agent 的 TreeRollout，返回所有结果

        Args:
            agent: 当前 agent
            node: 当前节点（包含刚执行的 step）

        Returns:
            Tuple[List[Any], str] | None: (Sub-agent 的所有 final answers, sub_agent_name)，如果没有则返回 None
        """
        # 检查当前 step 是否有 code_action
        if not hasattr(node.step, 'code_action') or not node.step.code_action:
            return None

        # 检测代码中是否调用了 managed_agents
        agent_calls = agent._extract_agent_calls(node.step.code_action)

        if not agent_calls:
            return None

        # 有 sub-agent 调用！
        for agent_call in agent_calls:
            sub_agent_name = agent_call['name']
            if sub_agent_name not in agent._original_managed_agents:
                continue

            sub_agent = agent._original_managed_agents[sub_agent_name]

            # 检查 sub-agent 是否需要分支
            if isinstance(sub_agent, BranchingCodeAgent) and sub_agent.branch_config.mode != 'none':
                # 递归执行 sub-agent 的 TreeRollout
                sub_task = agent_call['args'].get('task', '')
                print(f"[DEBUG] Calling sub-agent '{sub_agent_name}' with task: {sub_task[:100]}...")
                print(f"[DEBUG] Full agent_call: {agent_call}")
                sub_rollout = TreeRollout(sub_agent)
                sub_tree_memory = await sub_rollout.rollout(sub_task)

                # 将 sub-tree 关联到当前节点
                node.add_sub_tree(node.depth, sub_tree_memory.root)

                # 提取所有叶子节点的 final answers
                leaf_nodes = sub_tree_memory.get_all_leaf_nodes()
                sub_results = []
                for leaf in leaf_nodes:
                    # 从叶子节点的 trajectory 中找 final answer
                    for step in reversed(leaf.get_trajectory()):
                        if isinstance(step, ActionStep) and step.is_final_answer:
                            sub_results.append(step.action_output)
                            break

                return (sub_results, sub_agent_name) if sub_results else None

        return None


# ============================================================================
# 测试示例
# ============================================================================

if __name__ == "__main__":
    """
    简单测试示例

    使用方法：
    python myCogeAgent.py
    """
#     print("=" * 60)
#     print("CodeAgent Tree Rollout - Test Example")
#     print("=" * 60)

#     # 示例：创建一个带 sub-agent 的 agent
#     # 注意：这需要实际的 model 和 tools，这里只是展示结构

#     print("\n✓ Phase 1: 数据结构已实现")
#     print("  - CodeAgentTreeNode: 树节点")
#     print("  - CodeAgentTreeMemory: 树管理器")
#     print("  - PendingSubAgentStep: 暂停状态")

#     print("\n✓ Phase 2: Agent 接口已实现")
#     print("  - _extract_agent_calls(): AST 解析")
#     print("  - step_async(): 执行一步并检测分支")
#     print("  - continue_step(): 用 sub-result 继续执行")

#     print("\n✓ Phase 3: TreeRollout 执行器已实现")
#     print("  - rollout(): 主流程")
#     print("  - _rollout_node(): 递归执行")
#     print("  - _handle_sub_agent_branching(): 分支处理")

#     print("\n" + "=" * 60)
#     print("核心工作流程：")
#     print("=" * 60)
#     print("""
# 1. TreeRollout.rollout(task)
#    ↓
# 2. agent.step_async(task)
#    - 生成代码
#    - 检测 sub-agent 调用
#    ↓
# 3. 如果检测到 sub-agent:
#    - 返回 PendingSubAgentStep
#    - 递归执行 sub-agent 的 TreeRollout
#    - 为每个 sub-agent 分支创建主 agent 分支
#    - 用 continue_step() 继续执行
#    ↓
# 4. 构建完整的 CodeAgentTreeMemory
#    - 树状结构
#    - 每个节点包含 AgentMemory
#    - sub_trees 记录 sub-agent 调用
#     """)

#     print("\n" + "=" * 60)
#     print("使用示例代码：")
#     print("=" * 60)
#     print("""
# # 创建 sub-agent
# math_agent = BranchingCodeAgent(
#     name="math_agent",
#     tools=[calculator],
#     model=model,
#     branch_config=BranchConfig(mode='step_level', n_branches=2)
# )

# # 创建主 agent
# main_agent = BranchingCodeAgent(
#     name="main_agent",
#     tools=[...],
#     model=model,
#     managed_agents=[math_agent],
#     branch_config=BranchConfig(mode='agent_level')
# )

# # 执行 tree rollout
# rollout = TreeRollout(main_agent)
# tree_memory = rollout.rollout(task="Solve this problem")

# # 查看结果
# print(tree_memory.visualize())

# # 获取所有 trajectories
# all_trajs = tree_memory.get_all_trajectories()
# for memory in all_trajs:
#     print(f"Trajectory with {len(memory.steps)} steps")
#     """)

#     print("\n" + "=" * 60)
#     print("实现完成！ 🎉")
#     print("=" * 60)

    # ========================================================================
    # 实际运行示例（如果取消注释下面的代码）
    # ========================================================================
    """ """
    # 运行实际测试
    model = get_model()

    # 创建 sub-agent
    math_agent = BranchingCodeAgent(
        name="math_agent",
        tools=[],
        model=model,
        managed_agents=[],
        additional_authorized_imports=['sympy'],
        description="解决数学问题的专家,能使用 sympy 库进行精确的符号计算",
        max_steps=10,
        branch_config=BranchConfig(mode='agent_level')
    )

    # 创建主 agent
    main_agent = BranchingCodeAgent(
        name="main_agent",
        tools=[],
        model=model,
        managed_agents=[math_agent],
        # additional_authorized_imports=['sympy', 'math', 'numpy'],
        max_steps=3,
        branch_config=BranchConfig(mode='agent_level', n_branches=2),
        verbosity_level=1  # 增加 verbosity 来帮助 debugging
    )

    # 执行 tree rollout
    rollout = TreeRollout(main_agent)
    task = "使用 math_agent 来解决以下问题:已知椭圆G: x²/2 + y² = 1, 与x轴不重合的直线l经过左焦点F₁, 且与椭圆G相交于A,B两点, 弦AB的中点为M, 直线OM与椭圆G相交于C,D两点。\n(1)若直线l的斜率为1, 求直线OM的斜率\n(2)是否存在直线l, 使得|AM|² = |CM|·|DM|成立?若成立,求出直线l的方程;若不存在,请说明理由。"

    print(f"\n{'='*60}")
    print("开始执行 Tree Rollout (异步并行)...")
    print(f"Task: {task}")
    print(f"{'='*60}\n")

    # 使用 anyio.run 执行异步函数
    tree_memory = anyio.run(rollout.rollout, task)

    print(f"\n{'='*60}")
    print("执行完成！")
    print(f"{'='*60}\n")

    # 可视化树结构
    print("树结构：")
    print(tree_memory.visualize())

    # 保存树
    import os
    os.makedirs("tree_rollout_outputs", exist_ok=True)
    output_path = "tree_rollout_outputs/tree_memory.json"
    tree_memory.save(output_path)

    # 保存可视化文本
    viz_path = "tree_rollout_outputs/tree_visualization.txt"
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write(tree_memory.visualize())
    print(f"Tree visualization saved to: {viz_path}")

    # 获取所有 trajectories
    all_trajs = tree_memory.get_all_trajectories()
    print(f"\n总共生成了 {len(all_trajs)} 个 trajectories")

    final_answers = []
    for i, trajectory in enumerate(all_trajs):
        print(f"\nTrajectory {i+1}: {len(trajectory)} steps")

        # 找到 final answer
        final_answer = None
        for step in reversed(trajectory):
            if isinstance(step, ActionStep) and step.is_final_answer:
                final_answer = step.action_output
                break

        final_answers.append(final_answer)

        # 保存每个 trajectory
        traj_path = f"tree_rollout_outputs/trajectory_{i+1}.json"
        import json
        with open(traj_path, 'w', encoding='utf-8') as f:
            # trajectory 是 List[MemoryStep]，需要转换为 dict
            steps_dict = [step.dict() for step in trajectory]
            json.dump(steps_dict, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {traj_path}")
    print(f"\nFinal answers from all trajectories: {final_answers}")
    """
 """
