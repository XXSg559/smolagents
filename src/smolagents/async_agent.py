#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AsyncCodeAgent provides async execution capabilities for CodeAgent.

This module enables async/await patterns for smolagents CodeAgent instances,
allowing for concurrent execution and step-level coordination.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union

from .agents import CodeAgent
from .memory import AgentMemory

logger = logging.getLogger(__name__)


class AsyncCodeAgent(CodeAgent):
    """
    Async wrapper for CodeAgent that enables concurrent execution.

    This class maintains full compatibility with CodeAgent while adding
    async execution capabilities for step-level coordination and batch processing.
    """

    def __init__(
        self,
        # === CodeAgent 的原生参数，完全一致 ===
        tools: list,
        model,
        prompt_templates: Optional[Any] = None,
        additional_authorized_imports: Optional[list] = None,
        planning_interval: Optional[int] = None,
        executor_type: str = "local",
        executor_kwargs: Optional[dict] = None,
        max_print_outputs_length: Optional[int] = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        code_block_tags: Optional[Union[str, tuple]] = None,

        # === 异步新增参数 ===
        agent_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AsyncCodeAgent with same interface as CodeAgent.

        Args:
            All CodeAgent parameters (see CodeAgent documentation)
            agent_id: Optional unique identifier for this agent
            **kwargs: Additional CodeAgent parameters
        """
        # Initialize parent CodeAgent with all parameters
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            additional_authorized_imports=additional_authorized_imports,
            planning_interval=planning_interval,
            executor_type=executor_type,
            executor_kwargs=executor_kwargs,
            max_print_outputs_length=max_print_outputs_length,
            stream_outputs=stream_outputs,
            use_structured_outputs_internally=use_structured_outputs_internally,
            code_block_tags=code_block_tags,
            **kwargs
        )

        # Async-specific attributes
        self.agent_id = agent_id or f"async_agent_{uuid.uuid4().hex[:8]}"
        self.execution_lock = asyncio.Lock()

        logger.info(f"AsyncCodeAgent created: {self.agent_id}")

    async def run_async(self, task: str, **kwargs) -> Any:
        """
        Run a task asynchronously.

        Args:
            task: Task description to execute
            **kwargs: Additional parameters for execution

        Returns:
            Result from agent execution
        """
        async with self.execution_lock:
            # Run the synchronous run method in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.run(task, **kwargs)
            )
            return result

    async def step_async(self, **kwargs) -> Any:
        """
        Execute a single step asynchronously.

        Args:
            **kwargs: Parameters for step execution

        Returns:
            Result from step execution
        """
        async with self.execution_lock:
            # Run the synchronous step method in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.step(**kwargs)
            )
            return result

    async def execute_sequence_async(
        self,
        tasks: List[str],
        max_steps: int = 5
    ) -> Any:
        """
        Execute a sequence of tasks asynchronously.

        Args:
            tasks: List of task descriptions
            max_steps: Maximum steps per task

        Returns:
            Final result from sequence execution
        """
        logger.info(f"Agent {self.agent_id} executing {len(tasks)} tasks")

        result = None
        for i, task in enumerate(tasks):
            logger.debug(f"Agent {self.agent_id} executing task {i+1}/{len(tasks)}: {task[:50]}...")
            result = await self.run_async(task)

            # Add small delay between tasks to allow other agents to execute
            if i < len(tasks) - 1:
                await asyncio.sleep(0.01)

        return result

    def get_current_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state for synchronization.

        Returns:
            Dictionary representation of current memory state
        """
        if hasattr(self.memory, 'to_dict'):
            return self.memory.to_dict()
        else:
            # Fallback for basic memory representation
            return {
                "agent_id": self.agent_id,
                "memory_type": type(self.memory).__name__,
                "memory_length": len(getattr(self.memory, 'steps', []))
            }

    def __repr__(self) -> str:
        return f"AsyncCodeAgent(agent_id='{self.agent_id}', tools={len(self.tools)})"