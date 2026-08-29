from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


# ============================================================
# STATE OBJECTS
# ============================================================

@dataclass
class ModelSecurityState:
    model_name: str

    available: bool = True
    degraded: bool = False
    trusted: bool = True

    quarantined: bool = False

    candidate_available: bool = False
    candidate_evaluated: bool = False
    candidate_valid: bool = False

    deployed_trusted: bool = True
    validated: bool = True
    active: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class RecoveryAction:
    name: str
    model_name: str

    stage: str
    description: str

    estimated_runtime_sec: float

    requires_trusted_model: bool = False
    changes_security_state: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class RecoveryGoal:
    model_name: str

    trusted_required: bool = True
    validated_required: bool = True
    active_required: bool = True

    def satisfied(
        self,
        state: ModelSecurityState,
    ) -> bool:

        if (
            self.trusted_required
            and not state.trusted
        ):
            return False

        if (
            self.validated_required
            and not state.validated
        ):
            return False

        if (
            self.active_required
            and not state.active
        ):
            return False

        return True


# ============================================================
# RECOVERY ACTION CATALOG
#
# These are system-level recovery actions.
#
# They are intentionally more general than a scheduler's
# placement action.
# ============================================================

RECOVERY_STAGES = (
    "quarantine",
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
    "reactivate",
)


def default_recovery_actions(
    model_name: str,
    runtimes: Dict[str, float],
) -> List[RecoveryAction]:

    def rt(stage: str) -> float:
        return float(
            runtimes.get(
                stage,
                0.0,
            )
        )

    return [

        RecoveryAction(
            name=f"{model_name}:quarantine",
            model_name=model_name,
            stage="quarantine",
            description=(
                "Prevent an untrusted model from satisfying "
                "trusted workflow execution."
            ),
            estimated_runtime_sec=rt(
                "quarantine"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:update",
            model_name=model_name,
            stage="update",
            description=(
                "Create or restore a candidate model artifact."
            ),
            estimated_runtime_sec=rt(
                "update"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:evaluate",
            model_name=model_name,
            stage="evaluate",
            description=(
                "Evaluate the candidate model against the "
                "required validation criteria."
            ),
            estimated_runtime_sec=rt(
                "evaluate"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:package",
            model_name=model_name,
            stage="package",
            description=(
                "Prepare a validated candidate artifact "
                "for deployment."
            ),
            estimated_runtime_sec=rt(
                "package"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:deploy",
            model_name=model_name,
            stage="deploy",
            description=(
                "Deploy the candidate model artifact."
            ),
            estimated_runtime_sec=rt(
                "deploy"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:validate",
            model_name=model_name,
            stage="validate",
            description=(
                "Validate the deployed model in its target "
                "execution environment."
            ),
            estimated_runtime_sec=rt(
                "validate"
            ),
        ),

        RecoveryAction(
            name=f"{model_name}:reactivate",
            model_name=model_name,
            stage="reactivate",
            description=(
                "Return a validated trusted model to "
                "active workflow service."
            ),
            estimated_runtime_sec=rt(
                "reactivate"
            ),
        ),
    ]


# ============================================================
# ACTION PRECONDITIONS
# ============================================================

def action_allowed(
    action: RecoveryAction,
    state: ModelSecurityState,
) -> Tuple[bool, str]:

    stage = action.stage

    if stage == "quarantine":

        if state.trusted:
            return (
                False,
                "model_already_trusted",
            )

        if state.quarantined:
            return (
                False,
                "model_already_quarantined",
            )

        return (
            True,
            "ok",
        )


    if stage == "update":

        if not state.quarantined:
            return (
                False,
                "untrusted_model_not_quarantined",
            )

        return (
            True,
            "ok",
        )


    if stage == "evaluate":

        if not state.candidate_available:
            return (
                False,
                "no_candidate_available",
            )

        return (
            True,
            "ok",
        )


    if stage == "package":

        if not state.candidate_evaluated:
            return (
                False,
                "candidate_not_evaluated",
            )

        if not state.candidate_valid:
            return (
                False,
                "candidate_failed_evaluation",
            )

        return (
            True,
            "ok",
        )


    if stage == "deploy":

        if not state.candidate_valid:
            return (
                False,
                "candidate_not_valid",
            )

        return (
            True,
            "ok",
        )


    if stage == "validate":

        if not state.deployed_trusted:
            return (
                False,
                "no_deployed_candidate",
            )

        return (
            True,
            "ok",
        )


    if stage == "reactivate":

        if not state.validated:
            return (
                False,
                "deployed_model_not_validated",
            )

        return (
            True,
            "ok",
        )


    return (
        False,
        "unknown_action",
    )


# ============================================================
# ACTION EFFECTS
# ============================================================

def apply_action(
    action: RecoveryAction,
    state: ModelSecurityState,
) -> ModelSecurityState:

    allowed, reason = (
        action_allowed(
            action,
            state,
        )
    )

    if not allowed:
        raise ValueError(
            f"Action {action.name} not allowed: "
            f"{reason}"
        )

    stage = action.stage


    if stage == "quarantine":

        state.quarantined = True

        state.active = False

        state.deployed_trusted = False

        state.validated = False


    elif stage == "update":

        state.candidate_available = True

        state.candidate_evaluated = False

        state.candidate_valid = False


    elif stage == "evaluate":

        state.candidate_evaluated = True

        # Experiment 3 initially treats a completed evaluation
        # as producing a valid recovery candidate.
        #
        # Later we can replace this with empirical model-quality
        # validation if needed.
        state.candidate_valid = True


    elif stage == "package":

        # Packaging changes artifact readiness but does not
        # independently alter trust.
        pass


    elif stage == "deploy":

        state.deployed_trusted = True

        state.validated = False

        state.active = False


    elif stage == "validate":

        state.validated = True

        state.trusted = True

        state.degraded = False


    elif stage == "reactivate":

        state.active = True

        state.quarantined = False


    return state


# ============================================================
# SECURITY EVENT → RECOVERY WORLD
# ============================================================

def degraded_model_state(
    model_name: str,
) -> ModelSecurityState:

    return ModelSecurityState(
        model_name=model_name,

        available=True,

        degraded=True,

        trusted=False,

        quarantined=False,

        candidate_available=False,
        candidate_evaluated=False,
        candidate_valid=False,

        deployed_trusted=False,

        validated=False,

        active=True,
    )


def recovery_goal(
    model_name: str,
) -> RecoveryGoal:

    return RecoveryGoal(
        model_name=model_name,
        trusted_required=True,
        validated_required=True,
        active_required=True,
    )


# ============================================================
# AUDIT
# ============================================================

def state_summary(
    state: ModelSecurityState,
) -> str:

    return (
        f"model={state.model_name} "
        f"available={state.available} "
        f"degraded={state.degraded} "
        f"trusted={state.trusted} "
        f"quarantined={state.quarantined} "
        f"candidate_available={state.candidate_available} "
        f"candidate_evaluated={state.candidate_evaluated} "
        f"candidate_valid={state.candidate_valid} "
        f"deployed_trusted={state.deployed_trusted} "
        f"validated={state.validated} "
        f"active={state.active}"
    )
