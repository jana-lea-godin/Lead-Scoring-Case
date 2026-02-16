Gate-Definitionen klar benannt:
•	Gate 2: alpha = 0.05 (BH-FDR q-values)
•	Gate 3: min_stability_fraction = 0.80
•	Decision threshold: min_abs_lift = 0.03 (|OR−1| ≥ 0.03)


 
1) Executive Summary (1 Seite, copy-paste)

Kontext & Ziel
Wir wollen Conversion steigern – aber nur über Hebel, die tatsächlich wirken (kausal/strukturell), nicht über Proxy-Signale, die nur gut vorhersagen.
Ansatz: Dual Model + Decision Gates
•	Predictive Model (AUC 0.875): priorisiert Leads (“wer” konvertiert)
•	Structural Model (AUC 0.827): erklärt Treiber (“warum” konvertiert) – ohne process-nahe Variablen (z.B. Last Activity)
•	Gate 2 (Evidenz): Bootstrap + Benjamini–Hochberg q-values, Signifikanz bei q < 0.05
•	Gate 3 (Robustness → Entscheidung):
o	Effektgröße: abs_or_lift = |OR−1| ≥ 0.03
o	Stabilität: stability_sign ≥ 0.80
o	Regeln:
	SCALE = groß + signifikant + stabil
	INVESTIGATE = groß + (signifikant oder stabil)
	STOP = Rest

5 echte Conversion-Treiber (Structural “Truth”)

1.	Lead Origin: Lead Add Form (OR ~ 12.9)
High-intent Entry; stärkster Treiber – aber Attribution/Selection-Bias prüfen.
2.	Lead Source: Welingak Website (OR ~ 6.4)
Hochwertiger Traffic/fit → Owned/Partner Source skalieren.
3.	Persona-Fit: Working Professional (OR ~ 6.4)
Klarer ICP → Segment/Message/Routing darauf optimieren.
4.	Engagement: Total Time Spent on Website (OR ~ 3.0) (unterbewertet)
Engagement ist echter Hebel → Content/UX/Nurture als Conversion-Programm.
5.	Lead Origin: Landing Page Submission (OR ~ 0.31)
LP-Volume ist häufig low-intent → LP-Qualifizierung/Offer neu designen.



3 überschätzte Features (Predictive “Illusions”)

•	Better Career Prospects: predictive stark, structural schwach → Story, kein Hebel.
•	Facebook Source: predictive “schlecht”, structural nicht belegt → Kanal ≠ Ursache (meist Targeting/Offer/Handling).
•	Lead Import: predictive stark, structural nicht robust genug → Prozess/Attribution-Proxy.

3 unterschätzte Features (Hidden Levers)

•	Total Time Spent on Website: structural stark, predictive quasi 0 → echter Hebel wird im predictive Modell überdeckt.
•	Lead Origin: API (negativ): structural stabil → separate Behandlung/Qualification Gate nötig.
•	Olark Chat: structural positiv → Friction-Reducer, sauber testen und instrumentieren.
Entscheidung
Wir verschieben Fokus von “mehr Leads” zu “mehr qualifizierte Leads”:
ICP + High-Intent Paths + Engagement skalieren; LP/API Streams restrukturieren; Proxy-Narrative nicht zur Strategie machen.
________________________________________



2) Decision Playbook (Scale / Investigate / Stop) – als Management Action Plan


SCALE (Budget + Ops hochfahren)
1) ICP “Working Professionals”
•	Budget: mehr Spend auf ICP-Targeting, Lookalikes, Keywords/placements mit beruflichem Intent
•	Sales: Fast-lane SLA, eigenes Script, priorisierte Follow-ups
•	Produkt/Message: ROI, schedule-fit, career upgrade
2) Welingak Website Source
•	Scale: SEO/Partner placements/dedicated LPs
•	Guardrail: Attribution/UTM hardening, damit nicht over-credited wird
3) Score-Segmente p80/p90/p95 (segment_playbook)
•	score ≥ p80 (20% der Leads, ~89% CR): Core revenue engine
•	score ≥ p90/p95: “conversion-certain” → sofortige Sales-Priorisierung
________________________________________



INVESTIGATE (groß, aber erst Ursache/Design prüfen)
1) Lead Add Form (OR ~ 12.9)
•	Risiko: Selection Bias / pre-qualified traffic / duplicates
•	Checks: Quellenmix hinter Form, Fraud/Dedupe, Funnel-step leakage
•	Entscheidung: behalten & optimieren, aber nur nach Audit voll skalieren
2) Landing Page Submission (Volumen riesig, Conversion niedrig)
•	Experimente:
o	Qualifying question / progressive profiling
o	Offer ändern (weniger freebie)
o	Validation / friction
o	eigener Follow-up Prozess für LP-Leads
3) API Leads (großes Volumen, schlechte Conversion)
•	Hypothese: unreife Leads / Importqualität / fehlende Intent-Signale
•	Maßnahme: separate Nurture Tracks + Qualification Gate + anderes Routing
4) Olark Chat
•	Testdesign: Trigger, Platzierung, Script, handover
•	KPI: incremental uplift vs selection
 


STOP (nicht als strategischen Hebel behandeln)

1) “Better Career Prospects” als Hauptstrategie
•	Nur als Copy/Segmentation nutzen, nicht als Budget-Hebel.
2) “Facebook killen” als Reflex
•	Erst Diagnose: Targeting/Offer/Lead handling.
•	Stop erst nach Incrementality Test oder structural evidence.
3) Score ≤ p20 Segment (20% Leads, ~3% CR)
•	Keine Sales-Zeit → nur low-cost Automation.
 



3) 3 Key Insights Slides (fertig zum Einfügen)

Slide 1 — “Predictive ≠ Actionable: Dual Model Setup”
•	Predictive Logit (AUC 0.875): priorisiert Leads (“wer”)
•	Structural Logit (AUC 0.827): isoliert echte Treiber (“warum”), ohne process/proximity Variablen
•	Warum zwei Modelle: Predictive optimiert Forecast, Structural optimiert Decisions
Slide 2 — “Truth vs Performance: What really drives conversion”
True drivers (Structural, robust):
•	Lead Add Form (high intent)
•	Welingak Website (high-quality source)
•	Working Professional (ICP)
•	Time on Site (engagement lever)
•	Landing Page Submission (low intent / needs redesign)
Predictive illusions (overestimated):
•	Better Career Prospects
•	Facebook
•	Lead Import
→ gute Vorhersage ≠ echter Hebel



Slide 3 — “Action Plan: Scale / Investigate / Stop”


Scale
•	ICP + high-intent sources + top score segments (p80+)
Investigate
•	Lead Add Form (audit), LP submission redesign, API routing, Chat experimentation
Stop
•	Motivation-Narratives als Budget-Hebel; low-score p20 als Sales-Target; Facebook pauschal stoppen

