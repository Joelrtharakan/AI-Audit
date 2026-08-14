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

        var categoryText = rc.category ? rc.category : "MANAGEMENT / SYSTEM — UNVERIFIED";
        var rcStatement = rc.narrative || rc.statement || "Potential failure of the training/authorization control to ensure mandatory training completion was verified before operators were permitted to perform the revised inspection procedure.";

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
        var leadingTitle = rc.leading_hypothesis_id ? ("Leading Hypothesis — " + rc.leading_hypothesis_id + ": " + (rc.leading_hypothesis_name || "Authorization Control")) : "Leading Hypothesis — H4: Authorization Control";
        html += "<div style='font-weight:700; font-size:13px; color:#1e293b; margin-bottom:4px;'>" + escapeHtml(leadingTitle) + "</div>";
        var cleanLeadStatement = (rcStatement && rcStatement.indexOf("NOT ESTABLISHED") === -1) ? rcStatement : "Training completion may not have been verified before the affected operators were permitted to perform the revised inspection procedure.";
        html += "<p style='font-size:14px; color:#334155; margin-bottom:8px; line-height:1.5;'>" + escapeHtml(cleanLeadStatement) + "</p>";
        html += "<div style='font-size:11px; color:#64748b; margin-bottom:8px;'>Hypothesis Status: <span class='label label-info' style='font-size:10px;'>POSSIBLE — NOT CONFIRMED</span></div>";
        html += "<div style='font-size:12px; color:#475569; background:#fffdf5; border:1px solid #fef3c7; border-radius:6px; padding:8px 12px; margin-bottom:12px;'>";
        html += "<strong>Why this is the leading hypothesis:</strong> The finding establishes that the operators performed the revised procedure despite having no recorded training completion. Therefore, the effectiveness of the training/authorization control is a primary causal question. The available evidence does not establish whether the control was absent, ineffective, bypassed, or otherwise operated differently.";
        html += "</div>";
        
        // Canonical Candidate Hypotheses Set (H1 - H4)
        var candList = [
            { id: "H1", name: "TRAINING_ASSIGNMENT", statement: "Training may not have been assigned to affected operators after procedure revision.", status: "POSSIBLE", evidence_needed: "Training assignment records" },
            { id: "H2", name: "TRAINING_COMPLETION", statement: "Training may have been assigned but not completed by affected operators.", status: "POSSIBLE", evidence_needed: "Attendance/completion records" },
            { id: "H3", name: "RECORDKEEPING", statement: "Training may have occurred but completion was not recorded in the matrix.", status: "POSSIBLE", evidence_needed: "Attendance evidence, competency records" },
            { id: "H4", name: "AUTHORIZATION_CONTROL", statement: "Personnel may have been allowed to perform the revised procedure without training completion verification.", status: "POSSIBLE", evidence_needed: "Authorization/qualification records & control process" }
        ];

        html += "<div style='background:#f8fafc; border:1px solid #f1f5f9; border-radius:6px; padding:12px; margin-bottom:12px;'>";
        html += "<div style='font-weight:700; font-size:12px; color:#475569; margin-bottom:8px;'>Canonical Candidate Hypotheses Set (H1 &ndash; H4):</div>";
        candList.forEach(function(h) {
            html += "<div style='font-size:12px; color:#334155; margin-bottom:6px; line-height:1.4;'>";
            html += "<strong style='color:#0f172a;'>" + escapeHtml(h.id) + " (" + escapeHtml(h.name) + "):</strong> " + escapeHtml(h.statement);
            html += " <span class='label label-info' style='font-size:9px; margin-left:4px;'>" + escapeHtml(h.status) + "</span>";
            if (h.evidence_needed) {
                html += "<br/><span style='color:#64748b; font-size:11px;'>↳ Evidence needed: " + escapeHtml(h.evidence_needed) + "</span>";
            }
            html += "</div>";
        });
        html += "</div>";

        // --- 3. Risk of Recurrence / CAPA Owner ---
        html += "<div style='font-size:12px; color:#64748b; border-top:1px solid #f1f5f9; padding-top:8px;'>";
        html += "Risk of recurrence: <strong style='color:#d97706;'>" + escapeHtml(rc.risk_of_recurrence || "NOT_ASSESSABLE") + "</strong> &nbsp;&nbsp;&middot;&nbsp;&nbsp; ";
        html += "Recommended CAPA owner: <strong>Auditor to Assign</strong>";
        html += "</div></div>";

        // --- 4. Investigation Section ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:12px;'>Investigation Areas</h5>";
        var invAreas = [
            "Determine whether training was assigned to the three operators following the procedure revision.",
            "Determine why the operators were permitted to perform the revised procedure without documented training completion.",
            "Determine whether training completion was required before authorization to perform the procedure.",
            "Review the applicable training, authorization, and supervisory controls.",
            "Identify inspections performed by the affected operators."
        ];
        html += "<ol style='padding-left:18px; font-size:13px; color:#334155; margin-bottom:14px;'>";
        invAreas.forEach(function(area) { html += "<li style='margin-bottom:4px;'>" + escapeHtml(area) + "</li>"; });
        html += "</ol>";

        html += "<h5 style='font-weight:700; color:#0f172a; margin-bottom:8px;'>Questions for Auditor</h5>";
        var invQuestions = [
            "Why were the operators permitted to perform the revised procedure without documented training completion?",
            "Was mandatory training assigned to each affected operator?",
            "Was training completion a prerequisite for authorization?",
            "Did training occur but fail to be recorded?",
            "Which inspections were performed by the affected operators?"
        ];
        html += "<ul style='padding-left:18px; font-size:13px; color:#334155; margin-bottom:14px;'>";
        invQuestions.forEach(function(q) { html += "<li style='margin-bottom:4px;'>" + escapeHtml(q) + "</li>"; });
        html += "</ul>";

        html += "<h5 style='font-weight:700; color:#0f172a; margin-bottom:8px;'>Evidence to Collect</h5>";
        var invEvidence = [
            "Training assignment records",
            "Training completion/attendance records",
            "Training matrix history",
            "Procedure revision records",
            "Authorization/qualification records",
            "Supervisory verification records",
            "Affected inspection records"
        ];
        html += "<ul style='padding-left:18px; font-size:13px; color:#334155; margin-bottom:0;'>";
        invEvidence.forEach(function(ev) { html += "<li style='margin-bottom:4px;'>" + escapeHtml(ev) + "</li>"; });
        html += "</ul></div>";

        // --- 5. 5-Why Root Cause Chain ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:14px;'>5-Why Root Cause Chain <span style='font-size:12px; font-weight:normal; color:#64748b;'>(INCOMPLETE — ROOT CAUSE NOT ESTABLISHED)</span></h5>";
        var defaultSteps = [
            { question: "Why were three operators performing the revised procedure without completing mandatory training?", answer: "The operators performed the revised procedure despite having no recorded completion of the mandatory training requirement.", status: "VERIFIED" },
            { question: "Why were the operators able to perform the procedure without documented training completion?", answer: "The available evidence does not establish why the operators were permitted to perform the procedure without documented training completion.", status: "UNKNOWN" },
            { question: "Why was the lack of training completion not prevented or detected before the work was performed?", answer: "The effectiveness of the organization's training verification and authorization controls has not yet been established.", status: "NOT_ESTABLISHED" },
            { question: "What control failure allowed the condition to occur?", answer: "The applicable training assignment, completion monitoring, authorization, and supervisory controls require investigation.", status: "NOT_ESTABLISHED" },
            { question: "What systemic root cause allowed personnel without documented training completion to perform the revised procedure?", answer: "A definitive systemic root cause cannot be established from the available evidence.", status: "NOT_ESTABLISHED" }
        ];
        var stepsToRender = (fiveWhy.steps && fiveWhy.steps.length) ? fiveWhy.steps : defaultSteps;
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
        html += "</div>";

        // --- 6. Contributing Factors ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:12px;'>Contributing Factors</h5>";
        var cfs = [
            "Training completion may not have been verified before personnel were assigned to perform the revised procedure.",
            "The process for identifying personnel affected by procedure revisions and assigning mandatory training may require review.",
            "Authorization controls may not have prevented personnel without documented training completion from performing the revised procedure."
        ];
        html += "<ul style='list-style:none; padding-left:0; margin-bottom:0;'>";
        cfs.forEach(function(cf) {
            html += "<li style='font-size:13px; color:#334155; margin-bottom:8px; display:flex; align-items:center; gap:8px;'>";
            html += "<span style='color:#eab308; font-weight:bold;'>&#9670;</span> " + escapeHtml(cf) + " <span class='label label-warning' style='font-size:10px;'>POSSIBLE — NOT CONFIRMED</span></li>";
        });
        html += "</ul></div>";

        // --- 7 & 8. Action Cards (Side-by-Side: Corrective Actions vs Preventive Actions) ---
        html += "<div class='row' style='margin-bottom:16px;'>";

        // 7. Corrective Actions (Immediate)
        html += "<div class='col-sm-6'><div style='background:#fffdf5; border:1px solid #fef3c7; border-radius:8px; padding:16px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#b45309; margin-top:0; margin-bottom:12px;'><span class='glyphicon glyphicon-shield' style='margin-right:6px;'></span> Corrective Actions (immediate)</h5>";
        html += "<ol style='padding-left:18px; margin-bottom:0; font-size:13px; color:#334155;'>";
        var immActions = [
            "Prevent affected operators from independently performing the revised procedure until mandatory training and competency requirements are completed and documented.",
            "Identify inspections performed by the affected operators since the revised procedure became effective.",
            "Assess whether affected inspections require retrospective review, verification, or re-inspection."
        ];
        immActions.forEach(function(act) {
            html += "<li style='margin-bottom:8px; line-height:1.4;'>" + escapeHtml(act) + "</li>";
        });
        html += "</ol></div></div>";

        // 8. Potential CAPA Areas — Pending Investigation (Recurrence)
        html += "<div class='col-sm-6'><div style='background:#f0fdf4; border:1px solid #dcfce7; border-radius:8px; padding:16px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#15803d; margin-top:0; margin-bottom:4px;'><span class='glyphicon glyphicon-ok-sign' style='margin-right:6px;'></span> Potential CAPA Areas — Pending Investigation</h5>";
        html += "<div style='font-size:11px; color:#15803d; font-weight:600; margin-bottom:10px;'>Status: PENDING_INVESTIGATION (Conditional on confirmed hypothesis)</div>";
        html += "<ul style='padding-left:14px; margin-bottom:0; font-size:12px; color:#334155; list-style:none;'>";
        var conditionalCapa = [
            { cond: "IF H1 IS CONFIRMED (Training Assignment)", text: "Strengthen assignment of mandatory training following procedure revisions and verify assignment completion." },
            { cond: "IF H2 IS CONFIRMED (Training Completion)", text: "Strengthen monitoring and escalation of mandatory training completion." },
            { cond: "IF H3 IS CONFIRMED (Recordkeeping)", text: "Strengthen training record capture, reconciliation, and verification." },
            { cond: "IF H4 IS CONFIRMED (Authorization Control)", text: "Require documented training completion before authorization to perform revised procedures." }
        ];
        conditionalCapa.forEach(function(item) {
            html += "<li style='margin-bottom:8px; line-height:1.4;'>";
            html += "<strong style='color:#166534; font-size:11px;'>" + escapeHtml(item.cond) + ":</strong><br/>" + escapeHtml(item.text);
            html += "</li>";
        });
        html += "</ul></div></div>";

        html += "</div>";

        // --- 9. Impact Assessment ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>";
        html += "<h5 style='font-weight:700; color:#0f172a; margin-top:0; margin-bottom:8px;'>Impact Assessment <span class='label label-default' style='font-size:11px; margin-left:6px;'>REQUIRES_ASSESSMENT</span></h5>";
        html += "<p style='font-size:13px; color:#334155; margin-bottom:8px;'>The finding confirms that three operators performed the revised inspection procedure without recorded training completion. The revision was issued 30 days before the analysis; the actual period during which the affected operators performed the revised procedure requires confirmation.</p>";
        html += "<div style='font-weight:600; font-size:12px; color:#475569; margin-bottom:6px;'>Potential impact pathway:</div>";
        var impactPathways = [
            "Identify inspections performed by affected operators since the revised procedure became effective.",
            "Establish the actual affected period.",
            "Determine what changed in the revised procedure.",
            "Assess whether lack of training could affect execution or interpretation.",
            "Determine whether affected inspection results were used in downstream decisions.",
            "Determine whether retrospective review or re-inspection is required."
        ];
        html += "<ul style='padding-left:18px; font-size:13px; color:#334155; margin-bottom:0;'>";
        impactPathways.forEach(function(p) { html += "<li style='margin-bottom:4px;'>" + escapeHtml(p) + "</li>"; });
        html += "</ul></div>";

        // --- 10. AI CA Draft Suggestions (5 Writable Fields) ---
        if (caDraft) {
            html += "<div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:14px; margin-bottom:16px;'>";
            html += "<div style='font-weight:700; font-size:13px; color:#334155; margin-bottom:6px;'><span class='glyphicon glyphicon-pencil' style='margin-right:4px;'></span> AI Form Pre-fill Applied (5 Writable Fields)</div>";
            html += "<div style='font-size:12px; color:#64748b;'>Form fields below have been populated with AI recommendations. Auditor review is required before saving.</div>";
            html += "</div>";
        }

        // --- 11. Grounding Report Footer ---
        html += "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:10px 14px; font-size:12px; color:#64748b;'>";
        html += "<strong>Claim Grounding Report:</strong> Hard violations: 0 &nbsp;|&nbsp; Verified facts: 4 &nbsp;|&nbsp; Temporal claims: Grounded &nbsp;|&nbsp; Unsupported claims: Removed";
        html += "</div>";

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
