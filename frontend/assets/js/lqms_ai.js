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

        var html = "<div style='font-family: inherit; color: #1e293b; background: #f8fafc; border-radius: 8px; padding: 18px;'>";

        // --- 1. Observation Quality / Confidence Top Bar ---
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:10px 14px;'>";
        html += "<div style='font-size:13px; font-weight:600; color:#0f172a;'>Observation Quality: <span class='label label-success'>" + escapeHtml(report.observation_quality || "SUFFICIENT") + "</span></div>";
        html += "<div style='font-size:12px; color:#64748b;'><strong>Confidence:</strong> <span class='label label-info'>Obs: " + escapeHtml(report.observation_confidence || "HIGH") + "</span> &nbsp; <span class='label label-warning'>RC: " + escapeHtml(report.root_cause_confidence || "LOW") + "</span> &nbsp; <span class='label label-primary'>Overall: " + escapeHtml(report.overall_confidence || "MEDIUM") + "</span></div>";
        html += "</div>";

        // --- 2. AI-Suggested Root-Cause Hypothesis Card ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>";
        html += "<span style='font-weight:700; font-size:15px; color:#0f172a;'><span class='glyphicon glyphicon-lamp' style='color:#eab308; margin-right:6px;'></span> AI-Suggested Root-Cause Hypothesis</span>";
        html += "<span class='label label-default' style='background:#e2e8f0; color:#334155; font-size:11px; font-weight:600; padding:4px 10px; border-radius:12px;'>" + escapeHtml(categoryText) + "</span>";
        html += "</div>";
        html += "<div style='font-size:12px; color:#64748b; margin-bottom:8px;'>Root Cause Status: <span class='label label-warning' style='font-size:10px;'>" + escapeHtml(rc.status || "NOT_ESTABLISHED") + "</span> &nbsp;&nbsp;&middot;&nbsp;&nbsp; Confidence: <span class='label label-default' style='font-size:10px;'>LOW</span></div>";
        var candList = rc.candidate_hypotheses || [];

        if (rc.leading_hypothesis && rc.status !== "NOT_ESTABLISHED") {
            html += "<div style='font-weight:700; font-size:13px; color:#1e293b; margin-bottom:4px;'>Leading Hypothesis</div>";
            html += "<p style='font-size:14px; color:#334155; margin-bottom:8px; line-height:1.5;'>" + escapeHtml(rc.leading_hypothesis) + "</p>";
            html += "<div style='font-size:11px; color:#64748b; margin-bottom:12px;'>Hypothesis Status: <span class='label label-info' style='font-size:10px;'>POSSIBLE — NOT CONFIRMED</span></div>";
        } else if (candList.length) {
            html += "<div style='font-weight:700; font-size:13px; color:#1e293b; margin-bottom:4px;'>Candidate Root-Cause Hypotheses:</div>";
            html += "<p style='font-size:13px; color:#475569; margin-bottom:12px; font-style:italic;'>No leading hypothesis established from the available evidence. The following competing causal hypotheses require investigation:</p>";
        } else {
            html += "<p style='font-size:13px; color:#475569; margin-bottom:12px; font-style:italic;'>No leading hypothesis established from the available evidence, and no candidate hypotheses were generated for this finding. Auditor investigation is required.</p>";
        }

        if (candList.length) {
            html += "<div style='background:#f8fafc; border:1px solid #f1f5f9; border-radius:6px; padding:12px; margin-bottom:12px;'>";
            html += "<div style='font-weight:700; font-size:12px; color:#475569; margin-bottom:8px;'>Candidate Hypotheses:</div>";
            candList.forEach(function(h) {
                html += "<div style='font-size:12px; color:#334155; margin-bottom:10px; line-height:1.4; background:#fff; border:1px solid #e2e8f0; border-radius:5px; padding:8px 10px;'>";
                html += "<strong style='color:#0f172a;'>" + safeEsc(h.id) + " (" + safeEsc(h.name) + "):</strong> " + safeEsc(h.statement);
                html += " <span class='label label-info' style='font-size:9px; margin-left:4px;'>" + safeEsc(h.status) + "</span>";
                if (h.evidence_needed) {
                    html += "<br/><span style='color:#64748b; font-size:11px;'>↳ Evidence needed: " + safeEsc(h.evidence_needed) + "</span>";
                }
                if (h.discrimination_evidence) {
                    html += "<br/><span style='color:#7c3aed; font-size:11px;'>↳ Distinguishes from others: " + safeEsc(h.discrimination_evidence) + "</span>";
                }
                html += "</div>";
            });
            html += "</div>";
        }

        // --- 3. Risk of Recurrence / CAPA Owner ---
        html += "<div style='font-size:12px; color:#64748b; border-top:1px solid #f1f5f9; padding-top:8px;'>";
        html += "Risk of recurrence: <strong style='color:#d97706;'>" + escapeHtml(rc.risk_of_recurrence || "NOT_ASSESSABLE") + "</strong> &nbsp;&nbsp;&middot;&nbsp;&nbsp; ";
        html += "Recommended CAPA owner: <strong>Auditor to Assign</strong>";
        html += "</div></div>";

        // --- 4. Investigation Section (built from the actual investigation plan
        // and candidate hypotheses returned for THIS finding — never a fixed list) ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:12px;'>Investigation Areas</h5>";
        var invQuestions = (inv.questions && inv.questions.length) ? inv.questions : [];
        var invEvidence = (inv.evidence_to_collect && inv.evidence_to_collect.length) ? inv.evidence_to_collect : [];
        if (invQuestions.length) {
            html += "<ul style='padding-left:0; list-style:none; font-size:13px; color:#334155; margin-bottom:14px;'>";
            invQuestions.forEach(function(q, idx) {
                html += "<li style='margin-bottom:10px; background:#f8fafc; border:1px solid #f1f5f9; border-radius:6px; padding:10px 12px;'>";
                // Handle new structured {question, purpose, evidence} format
                if (q && typeof q === 'object') {
                    html += "<div style='font-weight:600; color:#0f172a; margin-bottom:4px;'>" + escapeHtml(q.question || '') + "</div>";
                    if (q.purpose && q.purpose !== 'not specified') {
                        html += "<div style='font-size:11px; color:#6366f1; margin-bottom:3px;'>↳ Resolves: " + escapeHtml(q.purpose) + "</div>";
                    }
                    if (q.evidence && q.evidence !== 'not specified') {
                        html += "<div style='font-size:11px; color:#64748b;'>↳ Evidence needed: " + escapeHtml(q.evidence) + "</div>";
                    }
                } else {
                    // Legacy plain string fallback
                    html += escapeHtml(q);
                    if (invEvidence[idx]) {
                        html += "<br/><span style='color:#64748b; font-size:11px;'>↳ Evidence: " + escapeHtml(invEvidence[idx]) + "</span>";
                    }
                }
                html += "</li>";
            });
            html += "</ul>";
        } else {
            html += "<p style='font-size:13px; color:#475569; font-style:italic; margin-bottom:14px;'>No investigation questions were generated for this finding.</p>";
        }
        if (invEvidence.length > invQuestions.length) {
            html += "<div style='font-size:12px; color:#64748b;'><strong>Additional evidence to collect:</strong> " + escapeHtml(invEvidence.slice(invQuestions.length).join("; ")) + "</div>";
        }
        html += "</div>";

        // --- 5. 5-Why Root Cause Chain (rendered only from what the backend actually
        // returned for THIS finding — no fabricated steps) ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:14px;'>5-Why Root Cause Chain <span style='font-size:12px; font-weight:normal; color:#64748b;'>(" + escapeHtml(fiveWhy.status_note || (fiveWhy.is_complete ? "COMPLETE" : "INCOMPLETE")) + ")</span></h5>";
        var stepsToRender = (fiveWhy.steps && fiveWhy.steps.length) ? fiveWhy.steps : [];
        if (stepsToRender.length) {
            $.each(stepsToRender, function(i, step) {
                var num = i + 1;
                var stStatus = (num === 1) ? "VERIFIED" : (step.status || "UNKNOWN");
                html += "<div style='display:flex; gap:12px; margin-bottom:12px; align-items:flex-start;'>";
                html += "<div style='background:#1e293b; color:#ffffff; font-weight:700; border-radius:4px; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0;'> " + num + " </div>";
                html += "<div style='flex-grow:1; background:#f8fafc; border:1px solid #f1f5f9; border-radius:6px; padding:10px 14px;'>";
                html += "<div style='font-weight:600; font-size:13px; color:#1e293b;'>" + escapeHtml(step.question) + "</div>";
                html += "<div style='font-size:13px; color:#475569; margin-top:4px;'>&#8627; " + escapeHtml(step.answer || "Requires verification") + " <span class='label label-default' style='font-size:10px; margin-left:6px;'>" + escapeHtml(stStatus) + "</span></div>";
                html += "</div></div>";
            });
        } else {
            html += "<p style='font-size:13px; color:#475569; font-style:italic;'>No 5-Why chain was generated for this finding.</p>";
        }
        if (rc.status === "NOT_ESTABLISHED") {
            html += "<div style='font-size:12px; font-weight:600; color:#b45309; background:#fffdf5; border:1px solid #fef3c7; border-radius:6px; padding:8px 12px; margin-top:8px;'>";
            html += "5-Why Status: INCOMPLETE &mdash; additional evidence required.";
            html += "</div>";
        }
        html += "</div>";

        // --- 6. Contributing Factors ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:12px;'>Contributing Factors</h5>";
        html += "<p style='font-size:13px; color:#64748b; font-style:italic; margin-bottom:0;'>Additional contributing factors are not established from the available evidence.</p>";
        html += "</div>";

        // --- 7 & 8. Action Cards (Side-by-Side: Corrective Actions vs Preventive Actions) ---
        html += "<div class='row' style='margin-bottom:16px;'>";

        // 7. Corrective Actions (Immediate) — from the CA draft's immediate_action
        // field for this finding, falling back to CAPA's recommended_investigation.
        html += "<div class='col-sm-6'><div style='background:#fffdf5; border:1px solid #fef3c7; border-radius:8px; padding:16px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#b45309; margin-top:0; margin-bottom:12px;'><span class='glyphicon glyphicon-shield' style='margin-right:6px;'></span> Corrective Actions (immediate)</h5>";
        var immActions = [];
        if (caDraft && caDraft.immediate_action) {
            immActions = String(caDraft.immediate_action).split(/\n+/).filter(Boolean);
        } else if (capa.recommended_investigation && capa.recommended_investigation.length) {
            immActions = capa.recommended_investigation;
        }
        if (immActions.length) {
            html += "<ol style='padding-left:18px; margin-bottom:0; font-size:13px; color:#334155;'>";
            immActions.forEach(function(act) {
                html += "<li style='margin-bottom:8px; line-height:1.4;'>" + escapeHtml(act) + "</li>";
            });
            html += "</ol>";
        } else {
            html += "<p style='font-size:13px; color:#475569; font-style:italic; margin-bottom:0;'>No immediate corrective action was generated for this finding.</p>";
        }
        html += "</div></div>";

        // 8. Potential CAPA Areas — Pending Investigation (Recurrence)
        html += "<div class='col-sm-6'><div style='background:#f0fdf4; border:1px solid #dcfce7; border-radius:8px; padding:16px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#15803d; margin-top:0; margin-bottom:4px;'><span class='glyphicon glyphicon-ok-sign' style='margin-right:6px;'></span> Potential CAPA Areas</h5>";
        html += "<div style='font-size:11px; color:#15803d; font-weight:600; margin-bottom:10px;'>Status: " + escapeHtml(capa.status || "INVESTIGATION_REQUIRED") + "</div>";
        var potentialAreas = capa.potential_areas || [];
        if (potentialAreas.length) {
            html += "<ul style='padding-left:14px; margin-bottom:0; font-size:12px; color:#334155; list-style:none;'>";
            potentialAreas.forEach(function(text) {
                html += "<li style='margin-bottom:8px; line-height:1.4;'>" + escapeHtml(text) + "</li>";
            });
            html += "</ul>";
        } else {
            html += "<p style='font-size:12px; color:#475569; font-style:italic; margin-bottom:0;'>No potential CAPA areas identified pending investigation.</p>";
        }
        html += "</div></div>";

        html += "</div>";

        // --- 8b. Conditional CAPA Actions (IF cause confirmed → THEN action) ---
        var conditionalActions = capa.conditional_actions || [];
        if (conditionalActions.length) {
            html += "<div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; padding:14px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
            html += "<h5 style='font-weight:700; color:#0369a1; margin-top:0; margin-bottom:8px;'><span class='glyphicon glyphicon-random' style='margin-right:6px;'></span>Conditional CAPA — Pending Investigation</h5>";
            html += "<div style='font-size:11px; color:#0369a1; margin-bottom:10px;'>These actions apply only if the stated cause is confirmed during investigation.</div>";
            conditionalActions.forEach(function(ca_action) {
                var ifText = safeStr(ca_action.if_cause_confirmed);
                var thenText = safeStr(ca_action.recommended_action);
                if (!ifText && !thenText) return;
                html += "<div style='background:#fff; border:1px solid #e0f2fe; border-radius:5px; padding:8px 12px; margin-bottom:8px; font-size:12px;'>";
                html += "<div style='color:#0369a1; font-weight:600; margin-bottom:3px;'>IF: " + escapeHtml(ifText) + "</div>";
                html += "<div style='color:#334155;'>→ THEN: " + escapeHtml(thenText) + "</div>";
                html += "</div>";
            });
            html += "</div>";
        }

        // --- 9. Impact Assessment (structured per Rule 17) ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:8px;'>Impact Assessment <span class='label label-default' style='font-size:11px; margin-left:6px;'>" + safeEsc(impact.status || "REQUIRES_ASSESSMENT") + "</span></h5>";
        if (impact.narrative) {
            html += "<p style='font-size:13px; color:#334155; margin-bottom:10px;'>" + safeEsc(impact.narrative) + "</p>";
        }
        // Render structured impact fields as a compact table when available
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
            html += "<table style='width:100%; font-size:12px; border-collapse:collapse; margin-bottom:8px;'>";
            impactFields.forEach(function(f) {
                html += "<tr>";
                html += "<td style='font-weight:600; color:#475569; width:130px; padding:4px 8px 4px 0; vertical-align:top;'>" + escapeHtml(f.label) + ":</td>";
                html += "<td style='color:#334155; padding:4px 0;'>" + safeEsc(f.val) + "</td>";
                html += "</tr>";
            });
            html += "</table>";
        }
        // Fallback: show areas list if no structured fields
        if (!impactFields.length) {
            var impactAreas = impact.areas || [];
            if (impactAreas.length) {
                html += "<div style='font-weight:600; font-size:12px; color:#475569; margin-bottom:6px;'>Potential impact pathway:</div>";
                html += "<ul style='padding-left:18px; font-size:13px; color:#334155; margin-bottom:0;'>";
                impactAreas.forEach(function(p) { html += "<li style='margin-bottom:4px;'>" + safeEsc(p) + "</li>"; });
                html += "</ul>";
            } else if (!impact.narrative) {
                html += "<p style='font-size:13px; color:#475569; font-style:italic; margin-bottom:0;'>No impact assessment was generated for this finding.</p>";
            }
        }
        html += "</div>";

        // --- 10. AI CA Draft Suggestions (5 Writable Fields) ---
        if (caDraft) {
            html += "<div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:14px; margin-bottom:16px;'>";
            html += "<div style='font-weight:700; font-size:13px; color:#334155; margin-bottom:6px;'><span class='glyphicon glyphicon-pencil' style='margin-right:4px;'></span> AI Form Pre-fill Applied (5 Writable Fields)</div>";
            html += "<div style='font-size:12px; color:#64748b;'>Form fields below have been populated with AI recommendations. Auditor review is required before saving.</div>";
            html += "</div>";
        }

        html += "<div style='text-align:center; font-size:11px; font-weight:600; color:#94a3b8; margin-top:16px; letter-spacing:0.5px;'>AI-GENERATED &mdash; HUMAN AUDITOR REVIEW REQUIRED</div>";
        html += "</div>";

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
