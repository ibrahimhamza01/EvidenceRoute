"""Validated budget profiles for two-year comparative execution."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedBudget, SharedLimits
from evidenceroute.research.rlm.controller import RootLimits


class ComparativeProfile(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    comparative_capture: bool = False
    max_groups: int = Field(ge=1)
    max_root_reinvestigation_calls: int = Field(default=2, ge=0)
    shared_limits: SharedLimits
    local_limits: RootLimits
    finalization: FinalizationPolicy

    @model_validator(mode="after")
    def validate_capacity(self):
        shared, local, policy = self.shared_limits, self.local_limits, self.finalization
        # Support one assignment per root decision and at least one retrieval per year.
        if shared.max_depth < 1 or shared.max_delegations < self.max_groups or shared.max_partitions < 2 * self.max_groups:
            raise ValueError("Shared delegation and partition limits must fit all requested two-year groups.")
        if shared.max_corpus_attempts < 2 * self.max_groups:
            raise ValueError("Shared corpus allowance must fit one retrieval per fiscal year per group.")
        if local.max_tool_calls < max(self.max_groups, 2) or local.max_iterations < self.max_groups + policy.finalization_attempts:
            raise ValueError("Local limits must fit initial assignments, child retrieval, and reserved finish decisions.")
        if self.comparative_capture and (
            policy.child_provider_attempts - policy.finalization_attempts < 3
            or local.max_iterations - policy.finalization_attempts < 3
            or local.max_tool_calls < 3
            or shared.max_corpus_attempts < 3 * self.max_groups
        ):
            raise ValueError("Comparative capture requires two retrieval turns and a capture turn before finalization.")
        budget = SharedBudget(shared, finalization=policy)
        with budget.invocation_allocation(root=True):
            if budget.active_allocation.investigation_left < self.max_groups:
                raise ValueError("Root provider allowance must fit one initial assignment per group.")
        return self


COMPARATIVE_PROFILES = {
    "diagnostic": ComparativeProfile(
        max_groups=2,
        shared_limits=SharedLimits(max_depth=1, max_delegations=2, max_partitions=4, max_provider_attempts=18,
                                   max_corpus_attempts=12, max_reported_tokens=100_000, deadline_seconds=180.0),
        local_limits=RootLimits(max_iterations=6, max_tool_calls=6, max_response_bytes=6_000, deadline_seconds=180.0),
        finalization=FinalizationPolicy(child_provider_attempts=6, finalization_attempts=2,
                                        finalization_tool_choice="required", root_token_headroom=12_000,
                                        child_token_headroom=12_000),
    ),
    "expanded": ComparativeProfile(
        max_groups=2,
        shared_limits=SharedLimits(max_depth=1, max_delegations=2, max_partitions=4, max_provider_attempts=36,
                                   max_corpus_attempts=30, max_reported_tokens=200_000, deadline_seconds=300.0),
        local_limits=RootLimits(max_iterations=12, max_tool_calls=12, max_response_bytes=16_000, deadline_seconds=300.0),
        finalization=FinalizationPolicy(child_provider_attempts=12, finalization_attempts=2,
                                        finalization_tool_choice="required", root_token_headroom=24_000,
                                        child_token_headroom=24_000, finalization_timeout_seconds=60.0,
                                        finalization_timeout_retries=1),
    ),
}


# Keep the diagnostic per-tree ceilings. The caller authorizes overall passes
# separately, including the number of batches and verification requests.
COMPARATIVE_PROFILES["full-corpus"] = ComparativeProfile.model_validate({
    **COMPARATIVE_PROFILES["diagnostic"].model_dump(),
    "comparative_capture": True,
    "max_root_reinvestigation_calls": 0,
})
