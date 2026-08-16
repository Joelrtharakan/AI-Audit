(function ($) {
    "use strict";

    var AI_BADGE_CLASS = "lqms-ai-suggested-badge";
    var AI_POPULATED_FLAG = "data-ai-populated";

    function config() {
        return window.LQMS_AI_CONFIG || {};
    }

    // Sprint 1 primary workflow: LLM-first, no RAG. Field names match
    // AnalyzeFindingRequest in the backend.
    function gatherAnalysisContext() {
        var text = (
            $("#lblDescription").val() ||
            $("#lblDescription").text() ||
            $("#txtFindingObsn").val() ||
            $("#txtFindingObsn").text() ||
            $("#txtFindingOrObsn").val() ||
            $("#txtFindingOrObsn").text() ||
            ""
        ).toString().trim();

        return {
            finding_text: text,
            department: $("#lblDepartments").text().trim(),
            branch: "",
            standard: $("#lblAuditCriteria").text().trim(),
            clause: $("#lblClauseNo").text().trim(),
            finding_type: $("#lblFindingType").text().trim(),
            nature_of_nc: $("#lblNatureOfNC").text().trim(),
            risk_severity: $("#lblRiskS").val() || $("#lblRiskS").text().trim(),
            risk_likelihood: $("#lblRiskL").val() || $("#lblRiskL").text().trim(),
            risk_result: $("#lblRiskResult").val() || $("#lblRiskResult").text().trim()
        };
    }

    // NOTE re: unicode escapes (Part 5.1) -- $.ajax with dataType:"json" parses the
    // response body via the browser's native JSON parser, which already turns \uXXXX
    // sequences into real characters (→, •, etc.). Nothing in this file re-serializes
    // already-parsed response text, so there is no place left for a literal \uXXXX
    // string to survive to the DOM. escapeHtml() below only HTML-escapes for safe
    // insertion; it does not touch unicode content.
    function escapeHtml(text) {
        return $("<div/>").text(text || "").html();
    }

    // Rule 21: Never show [object Object] — recursively convert any non-string value
    function safeStr(val) {
        if (val === null || val === undefined) return "";
        if (typeof val === "string") return val;
        if (typeof val === "boolean" || typeof val === "number") return String(val);
        if (Array.isArray(val)) return val.map(safeStr).filter(Boolean).join("; ");
        if (typeof val === "object") {
            // Try common text-bearing fields first
            return val.question || val.statement || val.description || val.text ||
                   val.narrative || val.action || val.recommended_action ||
                   val.if_cause_confirmed || JSON.stringify(val);
        }
        return String(val);
    }

    function safeEsc(val) { return escapeHtml(safeStr(val)); }

    function addAiBadge($field) {
        $field.attr(AI_POPULATED_FLAG, "1");
        var $existing = $field.next("." + AI_BADGE_CLASS);
        if ($existing.length) {
            return;
        }
        var $badge = $("<span/>", {
            "class": AI_BADGE_CLASS + " label label-info",
            text: "AI Suggested — Review Required",
            style: "margin-left:8px;font-weight:normal;"
        });
        $field.after($badge);
        $field.one("input change", function () {
            $field.removeAttr(AI_POPULATED_FLAG);
            $badge.remove();
        });
    }

    function setFieldValue(id, value) {
        var $field = $("#" + id);
        if (!$field.length || value === undefined || value === null || value === "") {
            return;
        }
        $field.val(value);
        addAiBadge($field);
    }

    function setRootCauseDropdown(category, otherText) {
        var $ddl = $("#ddlRootCauseType");
        if ($ddl.length && category) {
            $ddl.val(category);
            if (typeof $ddl.selectpicker === "function") {
                $ddl.selectpicker("refresh");
            }
            addAiBadge($ddl);
        }
        if (otherText) {
            setFieldValue("txtRootcauseCategory", otherText);
        }
    }

    function renderList(items) {
        if (!items || !items.length) {
            return "";
        }
        var html = "<ul>";
        $.each(items, function (i, item) {
            html += "<li>" + escapeHtml(item) + "</li>";
        });
        html += "</ul>";
        return html;
    }

    function riskBadgeKind(risk) {
        if (risk === "High") return "label-danger";
        if (risk === "Medium") return "label-warning";
        return "label-success";
    }

    function renderAnalysis(result) {
        var report = result.report || {};
        var caDraft = result.ca_draft || {};
        var rc = report.root_cause || {};
        var inv = report.investigation || {};
        var capa = report.capa || {};
        var impact = report.impact_assessment || {};
        var fiveWhy = report.five_why || {};

        var categoryText = rc.category ? rc.category : "TO_BE_CONFIRMED";
        var isDegraded = report.analysis_mode === "DEGRADED";

        var html = "<div style='font-family: \"Inter\", -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; color: #1e293b; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%); border-radius: 20px; padding: 28px; box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.08), 0 0 0 1px #cbd5e1; margin-top: 15px; margin-bottom: 25px;'>";

        // --- Header Banner ---
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; padding-bottom:20px; border-bottom:1px solid #cbd5e1;'>";
        html += "<div>";
        html += "<h3 style='margin:0; font-size:20px; font-weight:800; color:#0f172a; letter-spacing:-0.5px;'>AI Finding Investigation Analysis</h3>";
        html += "<div style='font-size:13px; color:#475569; margin-top:3px; font-weight:500;'>Autonomous Root Cause, Causal Inference & Audit Verification Engine</div>";
        html += "</div>";

        // Mode Pill & Provider Info
        html += "<div style='display:flex; align-items:center; gap:10px;'>";
        if (report.analysis_mode === "DETERMINISTIC") {
            html += "<span style='background:#f0f9ff; color:#0369a1; border:1px solid #bae6fd; font-size:11px; font-weight:700; padding:7px 16px; border-radius:30px; text-transform:uppercase; letter-spacing:0.8px; display:flex; align-items:center; gap:8px;'>";
            html += "<span style='width:8px; height:8px; border-radius:50%; background:#0284c7; display:inline-block; box-shadow:0 0 8px #0284c7;'></span> EVIDENCE-GROUNDED DETERMINISTIC ANALYSIS";
            html += "</span>";
        } else if (isDegraded) {
            html += "<span style='background:#fef2f2; color:#dc2626; border:1px solid #fecaca; font-size:11px; font-weight:700; padding:7px 16px; border-radius:30px; text-transform:uppercase; letter-spacing:0.8px; display:flex; align-items:center; gap:8px;'>";
            html += "<span style='width:8px; height:8px; border-radius:50%; background:#dc2626; display:inline-block; box-shadow:0 0 8px #dc2626;'></span> SAFETY FALLBACK MODE";
            html += "</span>";
        } else {
            html += "<span style='background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; font-size:11px; font-weight:700; padding:7px 16px; border-radius:30px; text-transform:uppercase; letter-spacing:0.8px; display:flex; align-items:center; gap:8px;'>";
            html += "<span style='width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block; box-shadow:0 0 8px #10b981;'></span> DEEP LLM SYNTHESIS ACTIVE";
            html += "</span>";
        }
        if (report.provider_used) {
            html += "<span style='background:#ffffff; color:#334155; border:1px solid #cbd5e1; font-size:11px; padding:7px 14px; border-radius:30px; font-weight:600; letter-spacing:0.3px; box-shadow:0 2px 4px rgba(0,0,0,0.03);'>";
            html += "Provider: <strong style='color:#0284c7;'>" + escapeHtml(report.provider_used.toUpperCase()) + "</strong>";
            if (report.fallback_used) html += " <span style='color:#b45309;'>(Fallback)</span>";
            html += "</span>";
        }
        html += "</div>";
        html += "</div>";

        // --- 1. Top Metrics Bar ---
        html += "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin-bottom:24px;'>";
        
        // Quality Metric Card
        var qColor = (report.observation_quality === "SUFFICIENT") ? "#059669" : "#d97706";
        var qBg = (report.observation_quality === "SUFFICIENT") ? "#ecfdf5" : "#fffbe6";
        var qBorder = (report.observation_quality === "SUFFICIENT") ? "#a7f3d0" : "#ffe58f";
        html += "<div style='background:#ffffff; border:1px solid #cbd5e1; border-radius:14px; padding:16px 18px; box-shadow:0 2px 6px rgba(15,23,42,0.04);'>";
        html += "<div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>Observation Quality</div>";
        html += "<div style='margin-top:8px;'>";
        html += "<span style='background:" + qBg + "; color:" + qColor + "; border:1px solid " + qBorder + "; font-size:13px; font-weight:800; padding:4px 12px; border-radius:8px; text-transform:uppercase; letter-spacing:0.5px;'>";
        html += escapeHtml(report.observation_quality || "SUFFICIENT");
        html += "</span>";
        html += "</div>";
        html += "</div>";

        // Confidence Metric Card
        html += "<div style='background:#ffffff; border:1px solid #cbd5e1; border-radius:14px; padding:16px 18px; box-shadow:0 2px 6px rgba(15,23,42,0.04);'>";
        html += "<div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>Analysis Confidence</div>";
        html += "<div style='display:flex; gap:6px; margin-top:8px;'>";
        html += "<span style='background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; font-size:10px; font-weight:700; padding:4px 8px; border-radius:6px;'>Obs: " + escapeHtml(report.observation_confidence || "HIGH") + "</span>";
        html += "<span style='background:#fffbe6; color:#b45309; border:1px solid #fde68a; font-size:10px; font-weight:700; padding:4px 8px; border-radius:6px;'>RC: " + escapeHtml(report.root_cause_confidence || "LOW") + "</span>";
        html += "<span style='background:#faf5ff; color:#7e22ce; border:1px solid #e9d5ff; font-size:10px; font-weight:700; padding:4px 8px; border-radius:6px;'>Overall: " + escapeHtml(report.overall_confidence || "MEDIUM") + "</span>";
        html += "</div>";
        html += "</div>";

        // Root Cause Category Card
        html += "<div style='background:#ffffff; border:1px solid #cbd5e1; border-radius:14px; padding:16px 18px; box-shadow:0 2px 6px rgba(15,23,42,0.04);'>";
        html += "<div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>6M Root Cause Taxonomy</div>";
        html += "<div style='font-size:16px; font-weight:800; color:#0284c7; margin-top:6px; letter-spacing:-0.2px;'>";
        html += escapeHtml(categoryText);
        html += "</div>";
        html += "</div>";

        html += "</div>"; // End metrics grid

        // ONE canonical root-cause status display (Section 8: never render
        // root-cause status in more than one place) — a compact STATUS / WHY
        // / LEADING HYPOTHESIS strip directly under the metrics grid. This
        // is the single place report.root_cause.status is rendered anywhere
        // on the page.
        var rcStatColor = (rc.status === "VERIFIED" || rc.status === "SUPPORTED") ? "#059669" : "#d97706";
        var rcWhy = rc.root_cause_basis || rc.narrative || "";
        var rcLeadDisplay = rc.leading_hypothesis || "";
        var rcLeadIsNone = !rcLeadDisplay || rcLeadDisplay.indexOf("NONE") === 0 || rcLeadDisplay.indexOf("TIED") === 0 || rc.leading_hypothesis_status === "TIED" || rc.leading_hypothesis_status === "NONE";
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 20px; margin-bottom:20px; box-shadow:0 2px 6px rgba(15,23,42,0.04);'>";
        html += "<div style='display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start;'>";
        html += "<div style='min-width:160px;'><div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>Root Cause Status</div>";
        html += "<div style='font-size:15px; font-weight:800; color:" + rcStatColor + "; margin-top:4px;'>" + escapeHtml(rc.status || "NOT_ESTABLISHED") + "</div></div>";
        if (rcWhy) {
            html += "<div style='flex:1; min-width:240px;'><div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>Why</div>";
            html += "<div style='font-size:13px; color:#334155; margin-top:4px; line-height:1.5;'>" + safeEsc(rcWhy) + "</div></div>";
        }
        html += "<div style='min-width:140px;'><div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.8px;'>Leading Hypothesis</div>";
        html += "<div style='font-size:14px; font-weight:800; color:" + (rcLeadIsNone ? "#b45309" : "#15803d") + "; margin-top:4px;'>" + (rcLeadIsNone ? "NONE" : escapeHtml(rcLeadDisplay)) + "</div></div>";
        html += "</div></div>";

        // --- Main Content Modules Container ---
        html += "<div style='display:flex; flex-direction:column; gap:20px;'>";

        // --- 1b. Evidence Status & Claim Provenance Card ---
        var evClaims = report.evidence_claims || [];
        var evConflicts = report.evidence_conflicts || [];
        if (evClaims.length || evConflicts.length) {
            html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
            html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #f1f5f9;'>";
            html += "<span style='font-weight:800; font-size:16px; color:#0f172a; letter-spacing:-0.2px;'>Evidence Ledger & Claim Provenance</span>";
            if (evConflicts.length) {
                html += "<span style='background:#fef2f2; color:#dc2626; font-size:11px; font-weight:800; padding:5px 12px; border-radius:20px; border:1px solid #fecaca; text-transform:uppercase; letter-spacing:0.5px;'>⚠ " + evConflicts.length + " Conflict(s) Detected</span>";
            } else {
                html += "<span style='background:#ecfdf5; color:#059669; font-size:11px; font-weight:800; padding:5px 12px; border-radius:20px; border:1px solid #a7f3d0; text-transform:uppercase; letter-spacing:0.5px;'>Consistent Evidence</span>";
            }
            html += "</div>";

            if (evConflicts.length) {
                html += "<div style='display:flex; flex-direction:column; gap:10px; margin-bottom:16px;'>";
                evConflicts.forEach(function(cf) {
                    html += "<div style='background:#fff1f2; border:1px solid #fecdd3; border-left:4px solid #e11d48; border-radius:8px; padding:12px 16px;'>";
                    html += "<div style='font-weight:800; font-size:13px; color:#9f1239; margin-bottom:4px;'><span class='glyphicon glyphicon-exclamation-sign'></span> " + safeEsc(cf.conflict_id || 'CONFLICT') + ": " + safeEsc(cf.proposition || '') + "</div>";
                    html += "<div style='font-size:12px; color:#881337;'>Status: <strong style='color:#be123c;'>" + safeEsc(cf.status || 'UNRESOLVED') + "</strong> — Available evidence contains conflicting reports. Objective records required for resolution.</div>";
                    html += "</div>";
                });
                html += "</div>";
            }

            if (evClaims.length) {
                html += "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:10px;'>";
                evClaims.forEach(function(c) {
                    var isVer = (c.status === "VERIFIED");
                    var cBg = isVer ? "#f0fdf4" : "#f8fafc";
                    var cBrd = isVer ? "#bbf7d0" : "#e2e8f0";
                    var cTxt = isVer ? "#166534" : "#334155";
                    var badgeBg = isVer ? "#dcfce7" : "#f1f5f9";
                    var badgeTxt = isVer ? "#15803d" : "#475569";
                    html += "<div style='background:" + cBg + "; border:1px solid " + cBrd + "; border-radius:8px; padding:10px 14px; display:flex; justify-content:space-between; align-items:flex-start;'>";
                    html += "<div style='font-size:12px; color:" + cTxt + "; line-height:1.4;'><strong>" + safeEsc(c.claim_id || 'C') + ":</strong> " + safeEsc(c.text) + "</div>";
                    html += "<span style='background:" + badgeBg + "; color:" + badgeTxt + "; font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; text-transform:uppercase; margin-left:8px; flex-shrink:0;'>" + safeEsc(c.status) + "</span>";
                    html += "</div>";
                });
                html += "</div>";
            }
            html += "</div>";
        }

        // --- 2. AI-Suggested Root-Cause Hypothesis Card ---
        html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #f1f5f9;'>";
        html += "<span style='font-weight:800; font-size:16px; color:#0f172a; letter-spacing:-0.2px;'>Root Cause Hypotheses & Causal Analysis</span>";
        html += "<span style='background:#f0f9ff; color:#0284c7; font-size:11px; font-weight:700; padding:5px 14px; border-radius:20px; border:1px solid #bae6fd; text-transform:uppercase; letter-spacing:0.5px;'>" + escapeHtml(categoryText) + "</span>";
        html += "</div>";

        var candList = rc.candidate_hypotheses || [];
        var leadingHyp = rc.leading_hypothesis || "";

        if (leadingHyp && !leadingHyp.startsWith("NONE") && !leadingHyp.startsWith("TIED")) {
            html += "<div style='background:linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border:1px solid #86efac; border-left:5px solid #22c55e; border-radius:12px; padding:16px 18px; margin-bottom:16px; box-shadow:0 2px 4px rgba(34,197,94,0.05);'>";
            html += "<div style='font-weight:800; font-size:12px; color:#15803d; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; display:flex; align-items:center; gap:6px;'><span class='glyphicon glyphicon-ok-sign'></span> Current Leading Hypothesis</div>";
            html += "<p style='font-size:14px; font-weight:700; color:#14532d; margin-bottom:8px; line-height:1.5;'>" + escapeHtml(leadingHyp) + "</p>";
            html += "<div style='font-size:11px; color:#166534; font-weight:600;'>Status: <span style='background:#ffffff; color:#15803d; padding:3px 10px; border-radius:6px; border:1px solid #86efac; font-weight:800;'>POSSIBLE — REQUIRES OBJECTIVE VERIFICATION</span></div>";
            html += "</div>";
        } else if (candList.length) {
            html += "<div style='background:#fffbe6; border:1px solid #ffe58f; border-left:5px solid #d97706; border-radius:10px; padding:14px 18px; margin-bottom:16px;'>";
            html += "<div style='font-weight:800; font-size:12px; color:#b45309; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'><span class='glyphicon glyphicon-info-sign'></span> Leading Hypothesis: NONE (Competing Hypotheses Tied)</div>";
            html += "<p style='font-size:13px; color:#78350f; margin:0; font-weight:500; line-height:1.4;'>Available evidence does not definitively single out one cause. The following competing hypotheses require objective record verification:</p>";
            html += "</div>";
        } else {
            html += "<p style='font-size:13px; color:#64748b; margin-bottom:16px; font-style:italic;'>No hypotheses established from available evidence. Auditor investigation is required.</p>";
        }

        if (candList.length) {
            html += "<div style='display:flex; flex-direction:column; gap:12px; margin-bottom:16px;'>";
            candList.forEach(function(h) {
                html += "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-left:5px solid #3b82f6; border-radius:10px; padding:14px 18px; transition:all 0.2s;'>";
                html += "<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;'>";
                html += "<strong style='color:#0f172a; font-size:14px; font-weight:700; line-height:1.4;'>" + safeEsc(h.id) + " (" + safeEsc(h.name) + "): <span style='font-weight:500; color:#334155;'>" + safeEsc(h.statement) + "</span></strong>";
                html += "<span style='background:#eff6ff; color:#1d4ed8; border:1px solid #dbeafe; font-size:10px; font-weight:800; padding:3px 10px; border-radius:14px; text-transform:uppercase; letter-spacing:0.5px; flex-shrink:0; margin-left:12px;'>" + safeEsc(h.status) + "</span>";
                html += "</div>";
                if (h.evidence_needed) {
                    html += "<div style='font-size:12px; color:#475569; margin-top:6px; background:#ffffff; padding:8px 12px; border-radius:6px; border:1px solid #f1f5f9;'><span style='color:#2563eb; font-weight:700;'>↳ Evidence needed:</span> " + safeEsc(h.evidence_needed) + "</div>";
                }
                if (h.discrimination_evidence) {
                    html += "<div style='font-size:12px; color:#6b21a8; margin-top:4px; background:#faf5ff; padding:8px 12px; border-radius:6px; border:1px solid #f3e8ff;'><span style='font-weight:700; color:#7e22ce;'>↳ Discrimination criterion:</span> " + safeEsc(h.discrimination_evidence) + "</div>";
                }
                html += "</div>";
            });
            html += "</div>";
        }

        // Risk of Recurrence Footer
        html += "<div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1px solid #e2e8f0; padding:12px 18px; border-radius:10px; font-size:13px;'>";
        html += "<div><span style='color:#64748b; font-weight:500;'>Risk of Recurrence:</span> <strong style='color:#d97706; font-weight:800; margin-left:4px;'>" + escapeHtml(rc.risk_of_recurrence || "NOT_ASSESSABLE") + "</strong></div>";
        var capaOwnerText = (rc.status === "NOT_ESTABLISHED") ? "To be assigned after root-cause confirmation" : "Auditor to Assign";
        html += "<div><span style='color:#64748b; font-weight:500;'>Recommended CAPA Owner:</span> <strong style='color:#0f172a; font-weight:700; margin-left:4px;'>" + escapeHtml(capaOwnerText) + "</strong></div>";
        html += "</div>";
        html += "</div>";

        // --- 3. Investigation Areas & Questions Card ---
        html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
        html += "<div style='margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #f1f5f9;'>";
        html += "<h5 style='font-weight:800; font-size:16px; color:#0f172a; margin:0; letter-spacing:-0.2px;'>Targeted Audit Investigation Plan</h5>";
        html += "</div>";

        var invQuestions = (inv.questions && inv.questions.length) ? inv.questions : [];
        var invEvidence = (inv.evidence_to_collect && inv.evidence_to_collect.length) ? inv.evidence_to_collect : [];
        
        if (invQuestions.length) {
            html += "<div style='display:flex; flex-direction:column; gap:12px; margin-bottom:16px;'>";
            invQuestions.forEach(function(q, idx) {
                html += "<div style='background:#faf5ff; border:1px solid #f3e8ff; border-left:4px solid #9333ea; border-radius:10px; padding:14px 16px;'>";
                if (q && typeof q === 'object') {
                    html += "<div style='font-weight:700; font-size:14px; color:#4c1d95; margin-bottom:6px; line-height:1.4;'>" + (idx + 1) + ". " + escapeHtml(q.question || '') + "</div>";
                    if (q.purpose && q.purpose !== 'not specified') {
                        html += "<div style='font-size:12px; color:#6b21a8; margin-bottom:4px; background:#ffffff; padding:6px 10px; border-radius:6px; border:1px solid #f3e8ff;'>↳ <strong style='font-weight:700; color:#7e22ce;'>Resolves:</strong> " + escapeHtml(q.purpose) + "</div>";
                    }
                    if (q.evidence && q.evidence !== 'not specified') {
                        html += "<div style='font-size:12px; color:#475569; background:#ffffff; padding:6px 10px; border-radius:6px; border:1px solid #f1f5f9;'>↳ <strong style='font-weight:700; color:#334155;'>Evidence required:</strong> " + escapeHtml(q.evidence) + "</div>";
                    }
                    if (q.possible_outcomes && q.possible_outcomes.length) {
                        html += "<div style='font-size:12px; color:#475569; background:#ffffff; padding:6px 10px; border-radius:6px; border:1px solid #f1f5f9; margin-top:4px;'>↳ <strong style='font-weight:700; color:#334155;'>Possible outcomes:</strong><ul style='margin:4px 0 0 18px; padding:0;'>";
                        q.possible_outcomes.forEach(function(o) {
                            html += "<li style='margin-bottom:2px;'>" + escapeHtml(o) + "</li>";
                        });
                        html += "</ul></div>";
                    }
                } else {
                    html += "<div style='font-weight:700; font-size:14px; color:#4c1d95;'>" + (idx + 1) + ". " + escapeHtml(q) + "</div>";
                    if (invEvidence[idx]) {
                        html += "<div style='font-size:12px; color:#475569; margin-top:4px;'>↳ <strong>Evidence:</strong> " + escapeHtml(invEvidence[idx]) + "</div>";
                    }
                }
                html += "</div>";
            });
            html += "</div>";
        } else {
            html += "<p style='font-size:13px; color:#64748b; font-style:italic; margin-bottom:16px;'>No specific investigation questions generated.</p>";
        }

        if (invEvidence.length > invQuestions.length) {
            html += "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; font-size:12px; color:#475569;'>";
            html += "<strong style='color:#0f172a;'>Additional Evidence Artifacts to Collect:</strong> " + escapeHtml(invEvidence.slice(invQuestions.length).join("; "));
            html += "</div>";
        }
        html += "</div>";

        // --- 4. 5-Why Root Cause Chain Card ---
        html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; padding-bottom:12px; border-bottom:1px solid #f1f5f9;'>";
        html += "<h5 style='font-weight:800; font-size:16px; color:#0f172a; margin:0; letter-spacing:-0.2px;'>5-Why Non-Circular Causal Chain</h5>";
        html += "<span style='font-size:11px; font-weight:800; background:#f1f5f9; color:#334155; padding:5px 12px; border-radius:20px; border:1px solid #e2e8f0; letter-spacing:0.5px;'>" + escapeHtml(fiveWhy.status_note || (fiveWhy.is_complete ? "COMPLETE" : "INCOMPLETE")) + "</span>";
        html += "</div>";

        var stepsToRender = (fiveWhy.steps && fiveWhy.steps.length) ? fiveWhy.steps : [];
        if (stepsToRender.length) {
            html += "<div style='display:flex; flex-direction:column; gap:14px;'>";
            $.each(stepsToRender, function(i, step) {
                var num = i + 1;
                var stStatus = step.status || "UNKNOWN";
                var badgeBg = (stStatus === "VERIFIED") ? "#dcfce7" : (stStatus.indexOf("REPORTED") !== -1 ? "#fef3c7" : (stStatus.indexOf("CONFLICT") !== -1 ? "#ffe4e6" : "#f1f5f9"));
                var badgeTxt = (stStatus === "VERIFIED") ? "#15803d" : (stStatus.indexOf("REPORTED") !== -1 ? "#b45309" : (stStatus.indexOf("CONFLICT") !== -1 ? "#e11d48" : "#475569"));
                var badgeBorder = (stStatus === "VERIFIED") ? "#bbf7d0" : (stStatus.indexOf("REPORTED") !== -1 ? "#fde68a" : (stStatus.indexOf("CONFLICT") !== -1 ? "#fecdd3" : "#e2e8f0"));

                html += "<div style='display:flex; gap:16px; align-items:stretch;'>";
                html += "<div style='display:flex; flex-direction:column; align-items:center;'>";
                html += "<div style='background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color:#ffffff; font-weight:800; border-radius:50%; width:34px; height:34px; display:flex; align-items:center; justify-content:center; font-size:14px; box-shadow:0 4px 10px rgba(15,23,42,0.25); flex-shrink:0;'> " + num + " </div>";
                if (num < stepsToRender.length) html += "<div style='width:2px; flex-grow:1; background:#cbd5e1; margin-top:6px; margin-bottom:2px;'></div>";
                html += "</div>";
                html += "<div style='flex-grow:1; background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:14px 18px;'>";
                html += "<div style='font-weight:700; font-size:14px; color:#0f172a; line-height:1.4;'>" + escapeHtml(step.question) + "</div>";
                html += "<div style='font-size:13px; color:#334155; margin-top:8px; display:flex; align-items:center; justify-content:space-between; background:#ffffff; padding:10px 14px; border-radius:8px; border:1px solid #f1f5f9;'>";
                html += "<span style='font-weight:500;'>↳ " + escapeHtml(step.answer || "Requires verification") + "</span>";
                html += "<span style='background:" + badgeBg + "; color:" + badgeTxt + "; border:1px solid " + badgeBorder + "; font-size:10px; font-weight:800; padding:3px 10px; border-radius:12px; flex-shrink:0; margin-left:12px; text-transform:uppercase; letter-spacing:0.5px;'>" + escapeHtml(stStatus) + "</span>";
                html += "</div>";
                html += "</div></div>";
            });
            html += "</div>";
        } else {
            html += "<p style='font-size:13px; color:#64748b; font-style:italic;'>No 5-Why chain generated for this finding.</p>";
        }

        if (rc.status === "NOT_ESTABLISHED") {
            html += "<div style='font-size:12px; font-weight:700; color:#b45309; background:#fffbe6; border:1px solid #ffe58f; border-radius:10px; padding:12px 16px; margin-top:16px; display:flex; align-items:center; gap:10px;'>";
            html += "<span class='glyphicon glyphicon-info-sign' style='font-size:16px;'></span> 5-Why Chain Status: INCOMPLETE — Chain safely stopped at evidence boundary to prevent circular reasoning.";
            html += "</div>";
        }
        html += "</div>";

        // --- 5. Side-by-Side Action Cards (Immediate Actions vs Potential CAPA) ---
        html += "<div class='row' style='margin-bottom:0;'>";

        // Immediate Corrective Actions
        html += "<div class='col-sm-6' style='margin-bottom:20px;'><div style='background:linear-gradient(180deg, #fffdf5 0%, #ffffff 100%); border:1px solid #fef3c7; border-radius:16px; padding:20px 22px; height:100%; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.05);'>";
        html += "<div style='margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #fef3c7;'>";
        html += "<h5 style='font-weight:800; color:#b45309; margin:0; font-size:16px; letter-spacing:-0.2px;'>Immediate Corrective Actions</h5>";
        html += "</div>";
        
        var immActions = [];
        if (caDraft && caDraft.immediate_action) {
            immActions = String(caDraft.immediate_action).split(/\n+/).filter(Boolean);
        } else if (capa.recommended_investigation && capa.recommended_investigation.length) {
            immActions = capa.recommended_investigation;
        }
        if (immActions.length) {
            html += "<ol style='padding-left:18px; margin-bottom:0; font-size:13px; color:#334155;'>";
            immActions.forEach(function(act) {
                html += "<li style='margin-bottom:10px; line-height:1.5; font-weight:500;'>" + escapeHtml(act) + "</li>";
            });
            html += "</ol>";
        } else {
            html += "<p style='font-size:13px; color:#64748b; font-style:italic; margin:0;'>No immediate action generated.</p>";
        }
        html += "</div></div>";

        // Investigation Areas / Potential CAPA Scope -- while root cause is
        // NOT_ESTABLISHED these are candidate areas to investigate, not
        // confirmed causes, so the heading must not imply a cause has
        // already been identified (Section 9).
        var areasNotEstablished = (rc.status === "NOT_ESTABLISHED");
        var areasHeading = areasNotEstablished ? "Investigation Areas — Cause Not Yet Established" : "Potential CAPA Scope";
        html += "<div class='col-sm-6' style='margin-bottom:20px;'><div style='background:linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%); border:1px solid #dcfce7; border-radius:16px; padding:20px 22px; height:100%; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.05);'>";
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #dcfce7;'>";
        html += "<h5 style='font-weight:800; color:#15803d; margin:0; font-size:16px; letter-spacing:-0.2px;'>" + areasHeading + "</h5>";
        html += "<span style='font-size:10px; font-weight:800; background:#dcfce7; color:#15803d; padding:4px 10px; border-radius:12px; border:1px solid #bbf7d0; text-transform:uppercase; letter-spacing:0.5px;'>" + escapeHtml(capa.status || "INVESTIGATION_REQUIRED") + "</span>";
        html += "</div>";

        var potentialAreas = capa.potential_areas || [];
        if (potentialAreas.length) {
            html += "<ul style='padding-left:0; list-style:none; margin-bottom:0; font-size:13px; color:#334155;'>";
            potentialAreas.forEach(function(text) {
                html += "<li style='margin-bottom:10px; display:flex; align-items:flex-start; gap:8px; font-weight:500;'>";
                html += "<span style='color:#22c55e; font-weight:800;'>✔</span> <span>" + escapeHtml(text) + "</span>";
                html += "</li>";
            });
            html += "</ul>";
        } else {
            html += "<p style='font-size:13px; color:#64748b; font-style:italic; margin:0;'>No CAPA areas identified pending investigation.</p>";
        }
        html += "</div></div>";

        html += "</div>"; // End row

        // --- 6. Conditional CAPA Actions ---
        var conditionalActions = capa.conditional_actions || [];
        if (conditionalActions.length) {
            html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
            html += "<div style='margin-bottom:6px;'>";
            html += "<h5 style='font-weight:800; color:#0369a1; margin:0; font-size:16px; letter-spacing:-0.2px;'>Conditional CAPA Action Branches</h5>";
            html += "</div>";
            html += "<div style='font-size:12px; color:#64748b; margin-bottom:14px; font-weight:500;'>These systemic actions are conditionally recommended pending confirmation during auditor investigation:</div>";
            
            html += "<div style='display:flex; flex-direction:column; gap:12px;'>";
            conditionalActions.forEach(function(ca_action) {
                var ifText = safeStr(ca_action.if_cause_confirmed);
                var thenText = safeStr(ca_action.recommended_action);
                if (!ifText && !thenText) return;
                html += "<div style='background:#f0f9ff; border:1px solid #bae6fd; border-left:5px solid #0284c7; border-radius:10px; padding:14px 16px;'>";
                html += "<div style='color:#0369a1; font-weight:800; font-size:12px; margin-bottom:6px; letter-spacing:0.3px;'>IF CONFIRMED: " + escapeHtml(ifText) + "</div>";
                html += "<div style='color:#0f172a; font-size:13px; font-weight:500;'>→ <strong style='font-weight:700; color:#0369a1;'>THEN ACTION:</strong> " + escapeHtml(thenText) + "</div>";
                html += "</div>";
            });
            html += "</div></div>";
        }

        // --- 7. Impact Assessment Card ---
        html += "<div style='background:#ffffff; border-radius:16px; padding:22px 24px; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.1); border:1px solid #e2e8f0;'>";
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #f1f5f9;'>";
        html += "<h5 style='font-weight:800; font-size:16px; color:#0f172a; margin:0; letter-spacing:-0.2px;'>Risk & Impact Assessment</h5>";
        html += "<span style='font-size:11px; font-weight:800; background:#fef2f2; color:#dc2626; border:1px solid #fecaca; padding:5px 12px; border-radius:20px; text-transform:uppercase; letter-spacing:0.5px;'>" + safeEsc(impact.status || "REQUIRES_ASSESSMENT") + "</span>";
        html += "</div>";

        if (impact.narrative) {
            html += "<p style='font-size:13px; color:#334155; margin-bottom:16px; line-height:1.6; background:#f8fafc; padding:12px 16px; border-radius:10px; border:1px solid #f1f5f9; font-weight:500;'>" + safeEsc(impact.narrative) + "</p>";
        }

        var impactFields = [
            { label: "Affected Object", val: impact.affected_object },
            { label: "Affected People", val: impact.affected_people },
            { label: "Affected Period", val: impact.affected_period },
            { label: "Process at Risk", val: impact.process_at_risk },
            { label: "Relevant Change", val: impact.relevant_change },
            { label: "Potential Effect", val: impact.potential_effect },
            { label: "Evidence Needed", val: impact.evidence_needed }
        ].filter(function(f) { return f.val && safeStr(f.val); });

        if (impactFields.length) {
            html += "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px;'>";
            impactFields.forEach(function(f) {
                html += "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px 14px;'>";
                html += "<div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:4px;'>" + escapeHtml(f.label) + "</div>";
                html += "<div style='font-size:13px; color:#0f172a; font-weight:600;'>" + safeEsc(f.val) + "</div>";
                html += "</div>";
            });
            html += "</div>";
        }
        html += "</div>";

        // --- 8. AI Form Pre-fill Applied Notice ---
        if (caDraft) {
            html += "<div style='background:linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border:1px solid #bfdbfe; border-radius:14px; padding:16px 20px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 4px 12px rgba(37,99,235,0.08);'>";
            html += "<div style='display:flex; align-items:center; gap:12px;'>";
            html += "<div style='background:#ffffff; border-radius:50%; width:32px; height:32px; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.05);'>";
            html += "<span class='glyphicon glyphicon-check' style='color:#2563eb; font-size:16px;'></span>";
            html += "</div>";
            html += "<div>";
            html += "<div style='font-weight:800; font-size:14px; color:#1e40af;'>5 Form Fields Pre-filled by AI</div>";
            html += "<div style='font-size:12px; color:#1e3a8a; font-weight:500;'>Immediate Action, Root Cause, Category, Preventive Action, & Impact populated below.</div>";
            html += "</div>";
            html += "</div>";
            html += "<span style='font-size:11px; font-weight:800; color:#2563eb; background:#ffffff; padding:6px 14px; border-radius:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05); border:1px solid #bfdbfe; text-transform:uppercase; letter-spacing:0.5px;'>Auditor Review Required</span>";
            html += "</div>";
        }

        html += "</div>"; // End main content modules container
        html += "<div style='text-align:center; font-size:11px; font-weight:800; color:#64748b; margin-top:20px; letter-spacing:1.5px;'>AI-GENERATED FINDING ANALYSIS &bull; HUMAN AUDITOR REVIEW REQUIRED</div>";
        html += "</div>"; // End main container

        return html;
    }

    function sanitizeFormText(val) {
        if (!val) return "";
        if (typeof val === "string") {
            var trimmed = val.trim();
            // If it looks like a serialized Python dict or JSON object, parse/clean it
            if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
                try {
                    // Replace Python single quotes with double quotes for JSON parsing
                    var jsonStr = trimmed.replace(/'/g, '"');
                    var parsed = JSON.parse(jsonStr);
                    return formatActionForForm(parsed);
                } catch (e) {
                    // Regex cleanup as fallback
                    return trimmed.replace(/'description':\s*/g, '')
                                  .replace(/'status':\s*'[^']*'/g, '')
                                  .replace(/{|}|\[|\]/g, '')
                                  .replace(/\s*,\s*/g, '\n')
                                  .trim();
                }
            }
            return trimmed;
        }
        return formatActionForForm(val);
    }

    function formatActionForForm(action) {
        if (!action) return "";
        if (typeof action === "string") return action;
        if (Array.isArray(action)) {
            return action
                .map(function(item) {
                    if (typeof item === "string") return item;
                    if (item && typeof item === "object") {
                        return item.description || item.statement || item.text || item.pathway || "";
                    }
                    return String(item || "");
                })
                .filter(Boolean)
                .map(function(item, idx) { return (idx + 1) + ". " + item; })
                .join("\n\n");
        }
        if (typeof action === "object") {
            return action.description || action.statement || action.text || action.pathway || JSON.stringify(action);
        }
        return String(action);
    }

    function applyAnalysisToForm(result) {
        var caDraft = result.ca_draft;
        if (!caDraft) return;

        setFieldValue("txtRootCause", sanitizeFormText(caDraft.immediate_action));
        setFieldValue("txtCorrectiveAction", sanitizeFormText(caDraft.root_cause));
        setFieldValue("txtPreventiveAction", sanitizeFormText(caDraft.preventive_action));
        setFieldValue("txtImpactAnalysis", sanitizeFormText(caDraft.impact_analysis));

        if (caDraft.root_cause_category) {
            setRootCauseDropdown(sanitizeFormText(caDraft.root_cause_category), "");
        }
    }

    function showPanel(html) {
        $("#lqmsAiResultsPanel").html(html);
        $("#lqmsAiResultsRow").show();
    }

    function setStatus(message, isError) {
        var $status = $("#capaAiStatus");
        $status.text(message || "");
        $status.css("color", isError ? "#a94442" : "#777");
    }

    function apiPost(path, payload) {
        var cfg = config();
        return $.ajax({
            url: cfg.apiBaseUrl + path,
            type: "POST",
            contentType: "application/json",
            dataType: "json",
            headers: { "X-Internal-Api-Key": cfg.internalApiKey || "" },
            data: JSON.stringify(payload),
            timeout: 300000 // 5 minutes for multi-step agent LLM graph execution
        });
    }

    $(function () {
        $("#lblDescription, #txtFindingObsn, #txtFindingOrObsn").on("input change", function () {
            if ($(this).val().trim().length > 0) {
                setStatus("", false);
            }
        });

        $("#btnAnalyzeFinding").on("click", function () {
            var $btn = $(this);
            var cfg = config();
            if (!cfg.apiBaseUrl) {
                setStatus("AI service not configured (missing apiBaseUrl in config.js).", true);
                return;
            }

            var context = gatherAnalysisContext();
            if (!context.finding_text) {
                setStatus("Enter a finding/observation before requesting AI assistance.", true);
                return;
            }

            $btn.prop("disabled", true);
            setStatus("Analyzing finding (running agent graph)…", false);

            apiPost("/api/v1/investigate", context)
                .done(function (result) {
                    showPanel(renderAnalysis(result));
                    applyAnalysisToForm(result);
                    setStatus("Investigation complete — please review before saving.", false);
                })
                .fail(function (jqXHR, textStatus) {
                    var message = "AI service unavailable. Existing form data has not been changed.";
                    if (textStatus === "timeout") {
                        message = "Investigation request timed out. The local LLM is taking longer to complete the graph — please click 'Suggest root cause & CAPA' to retry.";
                    } else if (jqXHR && jqXHR.responseJSON && jqXHR.responseJSON.detail) {
                        message = jqXHR.responseJSON.detail;
                    }
                    setStatus(message, true);
                })
                .always(function () {
                    $btn.prop("disabled", false);
                });
        });
    });
})(jQuery);
