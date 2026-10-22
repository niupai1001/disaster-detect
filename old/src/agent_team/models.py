from __future__ import annotations

DEFAULT_AGENTS = [
    {
        "agent_id": "research.lead",
        "department": "research",
        "role": "Research Lead",
        "capabilities": ["goal_research", "brainstorming", "adversarial_brainstorming", "research_handoff"],
        "allowed_actions": [
            "read_assignment",
            "write_brainstorm_note",
            "challenge_assumptions",
            "write_rebuttal",
            "write_research_memo",
            "emit_handoff",
        ],
        "state": "idle",
        "active_task_id": None,
        "active_assignment_id": None,
    },
    {
        "agent_id": "technical.lead",
        "department": "technical",
        "role": "Technical Lead",
        "capabilities": ["technical_brainstorming", "goal_interpretation", "technical_planning"],
        "allowed_actions": [
            "read_assignment",
            "write_brainstorm_note",
            "challenge_feasibility",
            "write_rebuttal",
            "write_technical_plan",
            "emit_handoff",
        ],
        "state": "idle",
        "active_task_id": None,
        "active_assignment_id": None,
    },
    {
        "agent_id": "implementation.dev",
        "department": "implementation",
        "role": "Implementer",
        "capabilities": ["code_changes", "implementation_notes"],
        "allowed_actions": ["read_assignment", "write_implementation_handoff"],
        "state": "idle",
        "active_task_id": None,
        "active_assignment_id": None,
    },
    {
        "agent_id": "qa.tester",
        "department": "qa",
        "role": "Tester",
        "capabilities": ["qa_brainstorming", "test_planning", "verification"],
        "allowed_actions": [
            "read_assignment",
            "write_brainstorm_note",
            "challenge_testability",
            "write_rebuttal",
            "write_test_report",
        ],
        "state": "idle",
        "active_task_id": None,
        "active_assignment_id": None,
    },
    {
        "agent_id": "review.lead",
        "department": "review",
        "role": "Reviewer",
        "capabilities": ["quality_review", "release_readiness"],
        "allowed_actions": ["read_assignment", "write_review_note"],
        "state": "idle",
        "active_task_id": None,
        "active_assignment_id": None,
    },
]

STAGE_AGENTS = {
    "research": "research.lead",
    "technical": "technical.lead",
    "implementation": "implementation.dev",
    "qa": "qa.tester",
    "review": "review.lead",
}

STAGE_PARTICIPANTS = {
    "brainstorm": ["research.lead", "technical.lead", "qa.tester"],
    "research": ["research.lead"],
    "technical": ["technical.lead"],
    "implementation": ["implementation.dev"],
    "qa": ["qa.tester"],
    "review": ["review.lead"],
}

STAGE_ARTIFACTS = {
    "brainstorm": "BrainstormMemo",
    "research": "ResearchMemo",
    "technical": "TechnicalPlan",
    "implementation": "ImplementationHandoff",
    "qa": "TestReport",
    "review": "ReviewNote",
}

BRAINSTORM_NOTE_ARTIFACTS = {
    "research.lead": "ResearchBrainstormNote",
    "technical.lead": "TechnicalBrainstormNote",
    "qa.tester": "QABrainstormNote",
}

STAGE_PREREQUISITES = {
    "technical": ["brainstorm"],
    "implementation": ["brainstorm", "technical"],
    "qa": ["brainstorm", "technical", "implementation"],
    "review": ["brainstorm", "technical", "implementation", "qa"],
}
