"""Blind Mutation & Cross-Domain Semantic Fidelity Evaluation Suite.

Tests generalized semantic fidelity across diverse domains without domain-specific rules:
1. Finance / Accounting (Invoice vs PO vs Payment vs Bank Reconciliation vs Recovery)
2. Pharma / Biotech (Calibration vs Maintenance vs Equipment Condition vs BMR Execution vs BMR Review)
3. Healthcare / Clinical (Prescription vs Dispensing vs Administration vs Administration Record)
4. Aviation / Maintenance (Work Order vs Authorization vs Physical Repair vs Sign-Off Log)
5. IT / Security (Access Request vs Credential Issuance vs Logon Event vs Audit Log)
6. Supply Chain / Logistics (Purchase Requisition vs Shipment vs Customs Clearance vs Receipt vs Inspection)
7. HR / Operations (Training Matrix vs Attendance Log vs Competency Assessment vs Authorization)
8. Environmental / Safety (Sensor Reading vs Threshold Limit vs Alarm Dispatch vs Operator Acknowledgement)
"""

import pytest

from app.agent.claim_extractor import extract_claims
from app.agent.invariants import evaluate_all_invariants
from app.agent.proposition_engine import build_propositions_from_ledger, build_semantic_graph
from app.models.agent import CanonicalFindingState, EvidenceStatus, InvestigateRequest
from app.services.semantic_subject import extract_semantic_subject, resolve_deviation


@pytest.mark.parametrize(
    "domain,finding_text,expected_subject,forbidden_concatenations",
    [
        (
            "Finance",
            "ERP system logged automated vendor payment dispatch of $45,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "payment",
            ["payment and approval", "invoice delivery and receipt", "approval execution"],
        ),
        (
            "Pharma",
            "During the internal audit of the cleanroom, analytical balance BAL-09 had valid calibration certificate on file, but daily balance check logs were missing for three consecutive shifts.",
            "daily balance check logs",
            ["calibration and check", "balance condition and record", "inspection execution"],
        ),
        (
            "Healthcare",
            "Pharmacy records confirmed prescription order dispensing at 08:00, but nursing staff stated the medication administration record was not signed prior to patient transfer.",
            "medication administration record",
            ["dispensing and administration", "prescription execution", "administration record delivery"],
        ),
        (
            "Aviation",
            "Hydraulic actuator work order WO-8812 was marked completed by maintenance personnel, but supervisory return-to-service authorization was absent from the technical logbook.",
            "supervisory return-to-service authorization",
            ["work order and authorization", "repair and sign-off", "maintenance execution"],
        ),
        (
            "IT Security",
            "Single Sign-On authentication server logged credential issuance for user ID USR-401, but the security analyst reported no multifactor authentication challenge was presented.",
            "multifactor authentication challenge",
            ["credential and authentication", "issuance and challenge", "access execution"],
        ),
        (
            "Supply Chain",
            "Logistics carrier manifest showed customs clearance and shipment arrival on container C-104, but warehouse receiving inspection records were not completed prior to putaway.",
            "warehouse receiving inspection records",
            ["shipment and inspection", "customs and receiving", "arrival execution"],
        ),
        (
            "HR Operations",
            "The training matrix required annual hazardous material handling certification, but employee attendance records for the refresher course were unavailable.",
            "employee attendance records",
            ["training matrix and attendance", "certification and authorization", "training execution"],
        ),
        (
            "Environmental",
            "Telemetry server recorded high-temperature threshold alarm dispatch at 14:32, but the control room log contained no operator acknowledgement record.",
            "operator acknowledgement record",
            ["alarm and acknowledgement", "sensor and dispatch", "alarm execution"],
        ),
    ],
)
def test_cross_domain_semantic_fidelity_and_atomicity(domain, finding_text, expected_subject, forbidden_concatenations):
    """Verify that semantic extraction isolates clean atomic entities without creating synthetic compound concepts."""
    resolved = resolve_deviation(finding_text)
    assert resolved.matched is True
    assert resolved.subject is not None
    assert resolved.affected_object is not None

    # Verify no synthetic compound concatenation
    for forbidden in forbidden_concatenations:
        assert forbidden.lower() not in resolved.affected_object.lower(), (
            f"[{domain}] Semantic collapse detected: '{forbidden}' found in affected_object '{resolved.affected_object}'"
        )
        assert forbidden.lower() not in resolved.subject.lower(), (
            f"[{domain}] Semantic collapse detected: '{forbidden}' found in subject '{resolved.subject}'"
        )

    # Decompose into claims, propositions, and semantic graph
    claims = extract_claims(finding_text)
    propositions = build_propositions_from_ledger(finding_text, claims)
    semantic_graph = build_semantic_graph(finding_text, claims, propositions)

    assert len(semantic_graph.nodes) > 0
    for node in semantic_graph.nodes:
        for forbidden in forbidden_concatenations:
            assert forbidden.lower() not in node.label.lower(), (
                f"[{domain}] Synthetic node created in SemanticGraph: '{node.label}' contains '{forbidden}'"
            )

    # Build canonical state and test invariants
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        finding_subject=resolved.subject,
        affected_object=resolved.affected_object,
        affected_process=resolved.affected_process or "Operational process",
        affected_activity=resolved.affected_activity or resolved.affected_object,
        observed_deviation=finding_text,
        deviation=finding_text,
        deviation_condition=resolved.condition or "nonconforming",
        evidence_claims=claims,
        propositions=propositions,
        semantic_graph=semantic_graph,
    )

    state = {
        "canonical_finding_state": canonical,
        "evidence_ledger": claims,
        "propositions": propositions,
    }

    is_valid, violations = evaluate_all_invariants(state)
    assert is_valid is True, f"[{domain}] Invariant violations detected: {violations}"


@pytest.mark.parametrize(
    "base_finding,mutated_finding,dimension,expected_stable_object",
    [
        (
            "ERP system logged automated vendor payment dispatch of $45,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "ERP system logged automated vendor payment dispatch of $90,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "QUANTITATIVE",
            "payment",
        ),
        (
            "ERP system logged automated vendor payment dispatch of $45,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "ERP system logged automated vendor payment dispatch of $45,000 on November 12, but the finance clerk reported that invoice approval was never completed.",
            "TEMPORAL",
            "payment",
        ),
        (
            "ERP system logged automated vendor payment dispatch of $45,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "Internal audit observed automated vendor payment dispatch of $45,000 on March 4, but the finance clerk reported that invoice approval was never completed.",
            "EPISTEMIC",
            "payment",
        ),
    ],
)
def test_single_property_mutation_invariance(base_finding, mutated_finding, dimension, expected_stable_object):
    """Verify that mutating a single property (e.g. quantity, date, or attribution) preserves the canonical affected entity."""
    base_res = resolve_deviation(base_finding)
    mut_res = resolve_deviation(mutated_finding)

    assert base_res.subject is not None
    assert mut_res.subject is not None
    assert expected_stable_object in base_res.subject.lower()
    assert expected_stable_object in mut_res.subject.lower()


@pytest.mark.parametrize(
    "domain,finding_text,expected_atomic_entity,forbidden_concepts",
    [
        (
            "Telecommunications",
            "Core network router CR-99 logged automated firmware patch deployment at 02:00, but the network operations engineer stated supervisory staging approval was not recorded.",
            "firmware patch",
            ["deployment and approval", "patch deployment (staging-approval)", "router staging execution"],
        ),
        (
            "Energy & Power",
            "Substation telemetry recorded automated transformer oil cooling valve actuation on August 15, but the field technician reported maintenance authorization sign-off was absent.",
            "transformer oil cooling valve",
            ["valve actuation and authorization", "actuation sign-off", "cooling maintenance execution"],
        ),
        (
            "Food & Beverage",
            "Automated pasteurization SCADA server recorded temperature cycle completion for vat V-12, but the quality assurance specialist stated pasteurization release verification was not performed.",
            "temperature cycle",
            ["cycle completion and verification", "temperature release execution", "vat verification control"],
        ),
        (
            "Automotive / Robotics",
            "Chassis robotic weld controller logged weld cycle execution on assembly line 3, but the welding inspector reported ultrasonic non-destructive testing record was missing.",
            "weld cycle",
            ["weld cycle and testing", "weld testing execution", "robotic inspection control"],
        ),
    ],
)
def test_unseen_cross_domain_semantic_invariance(domain, finding_text, expected_atomic_entity, forbidden_concepts):
    """Verify that unseen domain vocabularies preserve atomic entities without synthesizing compound concepts."""
    resolved = resolve_deviation(finding_text)
    assert resolved.matched is True
    assert resolved.subject is not None
    assert resolved.affected_object is not None

    for forbidden in forbidden_concepts:
        assert forbidden.lower() not in resolved.affected_object.lower(), (
            f"[{domain}] Semantic collapse: '{forbidden}' in affected_object '{resolved.affected_object}'"
        )
        assert forbidden.lower() not in resolved.subject.lower(), (
            f"[{domain}] Semantic collapse: '{forbidden}' in subject '{resolved.subject}'"
        )

    claims = extract_claims(finding_text)
    propositions = build_propositions_from_ledger(finding_text, claims)
    semantic_graph = build_semantic_graph(finding_text, claims, propositions)

    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        finding_subject=resolved.subject,
        affected_object=resolved.affected_object,
        affected_process=resolved.affected_process or "Operational process",
        affected_activity=resolved.affected_activity or resolved.affected_object,
        observed_deviation=finding_text,
        deviation=finding_text,
        deviation_condition=resolved.condition or "nonconforming",
        evidence_claims=claims,
        propositions=propositions,
        semantic_graph=semantic_graph,
    )

    state = {
        "canonical_finding_state": canonical,
        "evidence_ledger": claims,
        "propositions": propositions,
    }

    is_valid, violations = evaluate_all_invariants(state)
    assert is_valid is True, f"[{domain}] Invariant violations: {violations}"


def test_generated_text_cannot_mutate_canonical_state():
    """Verify that changing downstream investigation questions or 5-Why text has ZERO impact on canonical semantics."""
    from app.models.agent import InvestigationPlan, InvestigationQuestion, FiveWhyAnalysis, FiveWhyStep, CapaAnalysis

    finding = "The calibration certificate for balance BAL-014 expired on 2026-08-01, but the balance was used on 2026-08-05."
    resolved = resolve_deviation(finding)
    claims = extract_claims(finding)
    propositions = build_propositions_from_ledger(finding, claims)
    semantic_graph = build_semantic_graph(finding, claims, propositions)

    canonical = CanonicalFindingState(
        raw_finding=finding,
        observed_deviation=finding,
        finding_subject=resolved.subject,
        affected_object=resolved.affected_object,
        affected_process=resolved.affected_process or "Operational process",
        evidence_claims=claims,
        propositions=propositions,
        semantic_graph=semantic_graph,
    )

    initial_subject = canonical.finding_subject
    initial_object = canonical.affected_object

    # Mutate downstream investigation question wording arbitrarily
    mutated_plan = InvestigationPlan(
        questions=[
            InvestigationQuestion(
                question_id="Q_MUTATED_1",
                question="Why did the calibration management workflow experience a severe systemic failure?",
                purpose="Test downstream isolation",
                evidence="None",
            )
        ]
    )

    # Mutate downstream 5-Why wording arbitrarily
    mutated_five_why = FiveWhyAnalysis(
        steps=[
            FiveWhyStep(
                level=1,
                question="Why was the balance used?",
                answer="Because the calibration software failed to prevent operation.",
                status="UNKNOWN",
            )
        ]
    )

    # State check
    state = {
        "canonical_finding_state": canonical,
        "investigation_plan": mutated_plan,
        "five_why": mutated_five_why,
        "evidence_ledger": claims,
        "propositions": propositions,
    }

    is_valid, violations = evaluate_all_invariants(state)
    assert is_valid is True, f"Invariant failure: {violations}"

    # Verify canonical subject and affected object remained completely untouched
    assert canonical.finding_subject == initial_subject
    assert canonical.affected_object == initial_object
    assert "systemic failure" not in (canonical.affected_object or "").lower()
    assert "software" not in (canonical.affected_object or "").lower()


@pytest.mark.parametrize(
    "base_finding,mutated_finding,mutated_dimension",
    [
        (
            "Operator logged dispatch of batch B102 at 10:00, but supervisor reported verification check was absent.",
            "Technician logged dispatch of batch B102 at 10:00, but supervisor reported verification check was absent.",
            "ENTITY_ACTOR",
        ),
        (
            "Operator logged dispatch of batch B102 at 10:00, but supervisor reported verification check was absent.",
            "Operator logged dispatch of batch B102 at 16:30, but supervisor reported verification check was absent.",
            "TEMPORAL_TIME",
        ),
        (
            "Operator logged dispatch of 5 units of batch B102 at 10:00, but supervisor reported verification check was absent.",
            "Operator logged dispatch of 20 units of batch B102 at 10:00, but supervisor reported verification check was absent.",
            "QUANTITATIVE_COUNT",
        ),
    ],
)
def test_single_dimension_mutation_orthogonality(base_finding, mutated_finding, mutated_dimension):
    """Verify that mutating a single dimension (Actor, Time, Count) leaves the core object and relation structure invariant."""
    base_res = resolve_deviation(base_finding)
    mut_res = resolve_deviation(mutated_finding)

    assert base_res.matched is True
    assert mut_res.matched is True
    assert base_res.subject is not None
    assert mut_res.subject is not None

    if mutated_dimension == "ENTITY_ACTOR":
        assert base_res.actor != mut_res.actor
        assert base_res.subject == mut_res.subject
    elif mutated_dimension == "TEMPORAL_TIME":
        assert base_res.subject == mut_res.subject
    elif mutated_dimension == "QUANTITATIVE_COUNT":
        assert base_res.subject == mut_res.subject



