export const meta = {
  name: 'kb-slice-orchestrator',
  description: '專案自有 orchestrator（gh-native，零 beads）：plan(GitHub issues / 明確 slices) → 每 issue 並行 build(implement+review 同一 worktree) → 對抗驗證(嚴重度閘門+evidence) → 過閘門即開 PR → 獨立 Verdict agent 複驗 pushed SHA 並「自己」貼 verify/verdict status（貼的人=驗的人，過得了權限分類器；預設 Rung 2：人類只按 merge 鈕；autoMerge:true 升 Rung 3 — merge agent 預先武裝 gh pr merge --auto，checks 全綠 GitHub 自動合併；openPRs:false 退回 Rung 0 只備妥分支）',
  phases: [{ title: 'Plan' }, { title: 'Build' }, { title: 'OpenPRs' }, { title: 'Verdict' }],
}

// ─────────────────────────────────────────────────────────────────────────
// 為什麼有這一份（而不是直接用全域 slice-orchestrator）
// ─────────────────────────────────────────────────────────────────────────
// 1) 版控與抗漂移：全域 ~/.claude/workflows/slice-orchestrator.js 會被其他專案
//    或自動化（例：某夜的 obsidian lint 曾把 beads/bd init commit 進本 repo 的
//    main）改動；本檔隨 repo 版控，是本專案唯一權威的 orchestrator。
// 2) 接本專案自己的角色 prompt（project-docs/agents/{implement,review,merge}.md，
//    placeholder 契約是 {{ISSUE}}/{{BRANCH}}/{{TARGET_BRANCH}}）與本專案的
//    project-docs/CODING_STANDARD.md，而不是全域 agent-prompts。
// 3) 零 beads：本 repo 的 issue tracker 是 GitHub Issues + gh（見 CLAUDE.md）。
//    絕不呼叫任何 bd/beads 指令。護欄寫死在每個 sub-agent 的 prompt 裡。
//
// 執行方式（具名 workflow 不會自動解析 project-docs/ 下的檔，用 scriptPath 跑）：
//   Workflow({ scriptPath: "project-docs/agents/orchestrator.js",
//              args: { slices:[{id:"363",title:"..."}], standards:"project-docs/CODING_STANDARD.md" } })
//   slices 也吃裸 ID（[404] / ["404"]，自動正規化成 {id}；無效 id 直接 throw，不會派出 #undefined）
//   或依賴 gh 佇列： args: { only:["362","363"], skipPlan:true }
// ─────────────────────────────────────────────────────────────────────────

// args 容錯：物件 OR（誤傳的）JSON 字串都吃
const _argsObj = typeof args === 'string'
  ? (() => { try { return JSON.parse(args) } catch { return {} } })()
  : (args && typeof args === 'object' && !Array.isArray(args) ? args : {})

const CONFIG = {
  // 本專案自有的角色 prompt 與標準（repo-relative；build agent 在 worktree 全 checkout 內讀得到）
  promptsDir  : 'project-docs/agents',
  standards   : 'project-docs/CODING_STANDARD.md',
  branchPrefix: 'slice/',   // repo 慣例 slice/<N>-<desc>；orchestrator 自動產 slice/<N>，desc 可在 slices[].branch 指定
  baseBranch  : 'main',
  adversarial : true,       // 對抗驗證開（只擋 critical/high）
  openPRs     : true,       // Rung 2 預設（development-workflow.md auto-merge ladder；與 meta.description、結尾 note 一致）：過閘門即由 merge agent 開 PR，body 帶 Closes #N + verify 報告；人類仍負責按 merge。任何 agent 造成的 revert → 降回 Rung 0：傳 openPRs:false（只備妥分支）
  autoMerge   : false,      // Rung 3（opt-in，只在人類明確授權的 run 傳 true）：開 PR + 貼 verify/verdict status 後排 gh pr merge --auto --merge，required checks（CI 雙平台 + verify/verdict）全綠時由 GitHub 自動合併。前提照 ladder：verifier≠builder（本 workflow 天生滿足）、低風險 slice、絕不繞過任何 check。任何 agent 造成的 revert → 降回 Rung 1
  slices      : null,       // 明確切片清單 [{id,title,branch?}]；給了就用它當來源，完全不碰任何 tracker
  only        : null,       // 只做這些 id（gh 來源時當過濾；配 skipPlan 時當明確來源）
  skipPlan    : false,      // 配 only：跳過 gh 查詢，直接 build
  repo        : 'PaynePew/knowledge_base_qa_bot',
  ..._argsObj,
}
const MODELS = { plan:'haiku', build:'sonnet', verify:'opus', merge:'haiku', note:'haiku', ...(_argsObj.models ? _argsObj.models : {}) }

log(`effective config → slices=${CONFIG.slices ? CONFIG.slices.length : 'null'} · only=${JSON.stringify(CONFIG.only)} · adversarial=${CONFIG.adversarial} · openPRs=${CONFIG.openPRs} · autoMerge=${CONFIG.autoMerge} · base=${CONFIG.baseBranch} · standards=${CONFIG.standards}`)

const PLAN_SCHEMA = { type:'object', required:['issues'], properties:{
  issues:{ type:'array', items:{ type:'object', required:['id','branch'], properties:{
    id:{type:'string'}, title:{type:'string'}, type:{type:'string'}, branch:{type:'string'} } } } } }
const _BLOCKER_ITEMS = { type:'object', required:['severity','evidence'], properties:{
  file:{type:'string'}, line:{type:'number'}, issue:{type:'string'},
  evidence:{type:'string'},   // 必填：怎麼確認是真的（測試輸出/重現步驟/diff 具體行為）；逼 reviewer 拿證據，防「一定找得到東西」式的湊 blocker（與全域正本同構）
  severity:{ type:'string', enum:['critical','high','medium','low'] } } }
const VERDICT_SCHEMA = { type:'object', required:['verdict','blockers'], properties:{
  verdict:{ type:'string', enum:['pass','changes-requested'] },
  blockers:{ type:'array', items: _BLOCKER_ITEMS } } }
// Verdict stage 的回傳：判定 + 它實際貼上去的 status（sha/state），讓頂層能稽核「貼了什麼」
const VERDICT_POST_SCHEMA = { type:'object', required:['verdict','blockers','posted_sha','posted_state'], properties:{
  verdict:{ type:'string', enum:['pass','changes-requested'] },
  blockers:{ type:'array', items: _BLOCKER_ITEMS },
  posted_sha:{ type:'string' }, posted_state:{ type:'string', enum:['success','failure'] } } }

const mkBranch = (id) => CONFIG.branchPrefix + String(id).replace(/[^A-Za-z0-9._-]/g, '-')

// 防污染護欄 —— 每個會動檔案/git 的 sub-agent 都注入這段（堵住 bd init 落 main 的事件重演）
const GUARDRAILS = `硬性護欄（違反即視為失敗）：
- 這是 GitHub Issues + gh 的專案，絕不執行任何 beads / bd 指令（bd init/update/comment/ready/close/prime 一律禁止）。
- 絕不執行 git init、絕不初始化任何 issue tracker、絕不新增 .beads/.agents/.codex/AGENTS.md。
- 只 commit 你這片 slice 真正需要動的檔；絕不 commit 或改動 CLAUDE.md、.claude/settings.json、.gitignore、其他 slice 的成果、ADR/CONTEXT/CODING_STANDARD、既有測試。
- 絕不 push、絕不開 PR、絕不 merge、絕不 close issue（那是 merge 階段 / 人類的事）。`

// build：同一 worktree 內 從 baseBranch 開分支 → implement → 自我精簡（review.md）
const buildPrompt = (i) => `你在一個隔離 git worktree 內，獨自負責 GitHub issue #${i.id}（${i.title || 'untitled'}）。
重要：worktree 預設可能停在過時 base（origin/${CONFIG.baseBranch}），務必先從本地 ${CONFIG.baseBranch} 開分支，才帶得上前面已 merge 的依賴與 ADR。
1) 從 ${CONFIG.baseBranch} 開新分支：git switch -c ${i.branch} ${CONFIG.baseBranch}
2) 實作：讀 ${CONFIG.promptsDir}/implement.md 照做。代入 {{ISSUE}}=${i.id}、{{BRANCH}}=${i.branch}、{{TARGET_BRANCH}}=${CONFIG.baseBranch}。遵守與你所改檔案相關的 Accepted project-docs/adr/* 與 CONTEXT.md，以及 ${CONFIG.standards}。
3) 自我精簡：再讀 ${CONFIG.promptsDir}/review.md 照做（對你的改動 in-place 精簡並 commit）。
${GUARDRAILS}
回傳：改了哪些檔、git diff ${CONFIG.baseBranch}..${i.branch} 檔案清單、typecheck/test 是否全綠。`

// verify：獨立、唯讀；嚴重度標記，只有 critical/high 擋 merge
const verifyPrompt = (i) => `唯讀對抗式驗證 GitHub issue #${i.id}：執行 git diff ${CONFIG.baseBranch}..${i.branch}，找「影響正確性、安全性或明確需求」的 bug，並對照 ${CONFIG.standards}（特別是 §11 drift signals）與相關 Accepted project-docs/adr/* 檢查違規。
範圍紀律（precision over recall）：只回報有真實影響的 finding；風格偏好、假設性重構、與這片需求無關的改進建議一律不報。每個 blocker 必附 evidence——你怎麼確認它是真的（測試輸出、重現步驟、diff 中的具體行為）；給不出證據的疑慮最高只能標 medium。
若 diff 顯示它刪除/還原了既有成果（其他 slice、ADR、基礎設施），或新增了 .beads/.agents/AGENTS.md / 動了 CLAUDE.md / .claude 設定，標 critical（多半代表 base 拿錯或 agent 越界），evidence 寫出 diff 裡的具體檔案。
每個 blocker 標 severity：critical/high＝會壞/不安全/刪到別人成果/越界污染（擋 merge）；medium/low＝小毛病（回報不擋）。
不要改 code、不要 commit。回 verdict（pass＝無 critical/high；否則 changes-requested）與 blockers（file,line,issue,severity,evidence）。`

// merge（Rung 1 預設路徑）：讀專案 merge.md，push 分支 + 開 PR。merge.md 要求「從 issue 留言蒐集
// implementer/reviewer 報告」，但本 workflow 不貼 issue 留言 —— 改由此 prompt 內嵌報告餵給 merge agent，
// 填進 merge.md 既有的 PR body 段落（Closes #N + What was built + Reviewer verdict + concerns）。
const mergePrompt = (i, r) => {
  const nonBlocking = (r?.v?.blockers ?? []).filter(x => x.severity === 'medium' || x.severity === 'low')
  return `讀 ${CONFIG.promptsDir}/merge.md 照做：把分支 push 到 origin 並開 PR。代入 {{ISSUE}}=${i.id}、{{BRANCH}}=${i.branch}、{{TARGET_BRANCH}}=${CONFIG.baseBranch}。
merge.md 步驟 4 的「從 issue 留言蒐集報告」以下列內嵌報告代替（本 workflow 不貼 issue 留言），照模板填入 PR body 對應段落：
=== IMPLEMENTER（What was built / AC self-report 來源）===
${String(r?.build ?? '(無摘要)').slice(0, 2000)}
=== REVIEWER（Reviewer verdict / concerns 來源）===
verdict=${r?.v?.verdict ?? 'n/a'}；critical/high blocker 0 個（過閘門才會走到這步）；非阻擋 finding ${nonBlocking.length} 個：
${JSON.stringify(nonBlocking, null, 2).slice(0, 3000)}
=== /報告 ===
${GUARDRAILS}
- 你在共用的主工作樹操作：絕不執行 git stash / git reset / git checkout -- / git clean（主工作樹可能有頂層 session 的未提交工作）。發現未提交變更擋路時，一律 abort 回報，不要「幫忙清理」。你要 push 的分支在它自己的 worktree，主工作樹髒不髒與你的任務無關。
status 分工（取代 merge.md 步驟 3.5，2026-07-04 起）：你「絕不」貼任何 commit status —— verify/verdict 由後續的獨立 Verdict agent 複驗後自己貼（貼的人=驗的人，才不構成 self-approval；由你轉貼別人的 verdict 會被權限分類器整段擋下，實測過）。
${CONFIG.autoMerge
  ? `Rung 3（本 run 已獲人類明確授權 autoMerge）：開完 PR 後執行 gh pr merge <PR編號> --auto --merge —— 這只是預先武裝：required checks（CI 雙平台 + 獨立 Verdict agent 的 verify/verdict）全綠前 GitHub 不會合併，Verdict 貼 failure 就永遠鎖住。絕不 --admin、絕不繞過或停用任何 branch protection / check；若 --auto 排程失敗（例如 repo 未開 allow_auto_merge），回報原因並讓 PR 保持開啟，不要改用其他合併方式。絕不 gh issue close（merge 後 GitHub 依 Closes #N 自動關）。`
  : `注意：只 push + 開 PR（body 含 Closes #${i.id} + 上述 verify 報告）+ 在 issue 留 PR 連結；絕不自己按 merge、絕不 gh issue close（GitHub 會在人類 merge PR 時自動關）。`}
回傳：PR 編號與連結${CONFIG.autoMerge ? '（含是否已排入 auto-merge）' : ''}。`
}

// Verdict（Rung 2/3 的第二雙眼）：push 之後對 origin 上的 head SHA 做獨立複驗，並把「自己的」判定
// 貼成 commit status。關鍵約束：貼 status 的人必須就是作出判定的人 —— 由 merge agent（或頂層 session）
// 轉貼別人的 verdict 會被權限分類器判為 self-approval 而卡死（2026-07-04 實測；獨立 reviewer 自貼放行）。
const verdictPrompt = (i) => `你是獨立 reviewer，與本 run 的 build/merge agent 完全無關；對 GitHub issue #${i.id} 的已推分支做最終複驗，並把「你自己的」判定貼成 commit status。
1) git fetch origin ${i.branch}；sha = git rev-parse origin/${i.branch}；gh pr view 確認 PR 存在且 head 一致（PR 不存在 → 直接回報，不貼任何 status）。
2) 唯讀複驗 git diff ${CONFIG.baseBranch}..origin/${i.branch}：找「影響正確性、安全性或明確需求」的 bug，對照 ${CONFIG.standards}（特別是 §11 drift signals）與相關 Accepted project-docs/adr/*。範圍紀律同 verify（precision over recall；blocker 必附 evidence；給不出證據最高標 medium）；只有 critical/high 構成 changes-requested。
3) 把你的判定貼成該 SHA 的 status：gh api repos/${CONFIG.repo}/statuses/<sha> -f context=verify/verdict -f state=<pass→success；changes-requested→failure> -f description="independent verdict: <一句話摘要>"。你貼的是你自己剛作出的判定 —— 絕不代貼他人結論、絕不在 changes-requested 時貼 success、絕不貼到別的 SHA。
4) 你在共用主工作樹操作：唯讀（git fetch / rev-parse / diff / gh 可以）；絕不改 code、絕不 commit、絕不 git stash / reset / checkout -- / clean、絕不 merge、絕不關 PR/issue（合不合由 GitHub 依 required checks 決定）。
回傳：verdict、blockers、posted_sha、posted_state。`

// ── ① PLAN：明確 slices / only（零 tracker）優先；否則 gh issue list（GitHub 原生佇列）──
let todo
if (CONFIG.slices?.length) {
  // slices 收物件 [{id,...}]，裸 ID（數字/字串）也吃：不正規化的話 s.id 是 undefined，
  // 字串化成 "undefined" 後每個 build agent 只能拿著 #undefined 停手（2026-07-04 燒 240k tokens 的教訓）。
  const norm = CONFIG.slices.map(s => (s && typeof s === 'object') ? s : { id: s })
  const bad = norm.filter(s => s.id === undefined || s.id === null || String(s.id).trim() === '' || String(s.id) === 'undefined')
  if (bad.length) throw new Error(`slices 含無效 id：${JSON.stringify(bad)} — 傳 [{id,title,branch}] 物件或裸 ID（如 [404] / ["404"]）`)
  todo = norm.map(s => ({ id:String(s.id), title:s.title || '', type:s.type || 'task', branch: s.branch || mkBranch(s.id) }))
  log(`明確 slices：直接做指定的 ${todo.length} 片（不查任何 tracker）`)
} else if (CONFIG.skipPlan && CONFIG.only?.length) {
  todo = CONFIG.only.map(id => ({ id:String(id), title:'', type:'task', branch: mkBranch(id) }))
  log(`skipPlan：直接做指定的 ${todo.length} 個 id（不查任何 tracker）`)
} else {
  phase('Plan')
  const plan = await agent(
    `執行 gh issue list --repo ${CONFIG.repo} --state open --label ready-for-agent --limit 100 --json number,title,labels 取得目前「ready-for-agent」的 GitHub issue（不要自己推依賴）。
排除帶有 'epic' 或 'prd' label 的（那是 PRD/容器，不是可實作的切片）。
把每筆整理成 {id, title, type, branch}：id = issue number 的字串、branch = "${CONFIG.branchPrefix}" + id。只回傳 open、ready-for-agent 且非 epic 的。
若無法列 issue（無 remote / gh 未認證），回 {issues: []}（不要報錯）。`,
    { label: 'plan(gh issue list)', phase: 'Plan', schema: PLAN_SCHEMA, model: MODELS.plan })
  todo = (plan?.issues ?? []).filter(i => i.type !== 'epic')
  if (CONFIG.only) todo = todo.filter(i => CONFIG.only.map(String).includes(i.id))
}
if (!todo.length) { log('沒有可並行的 issue（傳 {slices:[...]} 或 {only:[...],skipPlan:true}，或先開 ready-for-agent 的 GitHub issue）'); return { planned: 0 } }
log(`要做 ${todo.length} 個 issue：${todo.map(i => i.title ? `${i.id}(${i.title})` : i.id).join(', ')}`)

// ── ② BUILD（每 issue 並行；build=implement+review 同 worktree、verify 獨立唯讀且嚴重度閘門）──
phase('Build')
const lbl = (i) => i.title ? `${i.id} · ${i.title}` : i.id
const built = await pipeline(
  todo,
  (issue)     => agent(buildPrompt(issue), { label:`build:${lbl(issue)}`, phase:'Build', isolation:'worktree', model: MODELS.build }),
  (b, issue)  => CONFIG.adversarial
    ? agent(verifyPrompt(issue), { label:`verify:${lbl(issue)}`, phase:'Build', schema: VERDICT_SCHEMA, model: MODELS.verify })
        .then(v => ({ build: b, v, blocking: (v?.blockers ?? []).filter(x => x.severity === 'critical' || x.severity === 'high') }))
    : { build: b, v: { verdict: 'skipped', blockers: [] }, blocking: [] },
)

// ── 確定性閘門：build 完成 ∧（未開驗證 或 無 critical/high blocker）──
const eligible = todo.map((issue, i) => ({ issue, r: built[i] }))
  .filter(x => x.r && (!CONFIG.adversarial || (x.r.v && x.r.blocking.length === 0)))
const okIds = new Set(eligible.map(e => e.issue.id))
const blocked = todo.filter(i => !okIds.has(i.id)).map(i => i.id)
log(`${eligible.length}/${todo.length} 過閘門${CONFIG.adversarial ? '（含對抗驗證，只擋 critical/high）' : ''}` +
    (blocked.length ? `；未過：${blocked.join(', ')}` : ''))

if (!eligible.length) return { planned: todo.length, opened: 0, blocked, note: '沒有 issue 過閘門' }
if (!CONFIG.openPRs)
  return { planned: todo.length, opened: 0, eligible: eligible.map(e => e.issue), blocked,
           note: `Rung 0 退回模式（openPRs:false）：分支已備妥並過閘門（從 ${CONFIG.baseBranch} 開）。由頂層 session 逐一：驗證真實 artifact → 開 PR(Closes #N) → CI 雙綠 → 人工 merge。` }

// ── ③ OPEN PRs → ④ VERDICT（pipeline，無 barrier：A 片在複驗時 B 片可能還在開 PR）──
// OpenPRs：push + 開 PR（autoMerge 時順手武裝 --auto，無害：checks 綠之前 GitHub 不會合）。
// Verdict：獨立 reviewer 對 pushed SHA 複驗並「自己」貼 verify/verdict —— Rung 2 從此不需要人貼 status，
// Rung 3（autoMerge:true）連 merge 鈕都不用按；Verdict 貼 failure 時 PR 被 branch protection 鎖死。
const prs = await pipeline(
  eligible,
  (e) => agent(mergePrompt(e.issue, e.r), { label:`openpr:${lbl(e.issue)}`, phase:'OpenPRs', model: MODELS.merge })
    .then(out => ({ e, id: e.issue.id, out })).catch(() => ({ e, id: e.issue.id, out: null })),
  (opened) => !opened?.out
    ? { id: opened?.e?.issue?.id, out: null, verdictPost: null }
    : agent(verdictPrompt(opened.e.issue), { label:`verdict:${lbl(opened.e.issue)}`, phase:'Verdict', schema: VERDICT_POST_SCHEMA, model: MODELS.verify })
        .then(vp => ({ id: opened.id, out: opened.out, verdictPost: vp }))
        .catch(() => ({ id: opened.id, out: opened.out, verdictPost: null })),
)
const failedVerdicts = prs.filter(p => p?.verdictPost?.posted_state === 'failure').map(p => p.id)
const missingVerdicts = prs.filter(p => p?.out && !p?.verdictPost).map(p => p.id)
return { planned: todo.length, opened: prs.filter(p => p?.out).length, blocked, failedVerdicts, missingVerdicts, prs,
         note: (CONFIG.autoMerge
           ? 'PR 已開 + --auto 已武裝 + 獨立 Verdict agent 已複驗自貼 verify/verdict（Rung 3：checks 全綠時 GitHub 自動合併）。頂層 session：驗證真實 artifact，並確認 auto-merge 實際發生。'
           : 'PR 已開 + verify/verdict 已由獨立 Verdict agent 複驗自貼（Rung 2）。頂層 session：驗證真實 artifact → CI 綠 → 人類只需按 merge。')
           + (failedVerdicts.length ? `；Verdict 貼了 failure（PR 鎖死待人裁）：${failedVerdicts.join(', ')}` : '')
           + (missingVerdicts.length ? `；Verdict agent 未完成（status 缺席、PR 不能合，要人補）：${missingVerdicts.join(', ')}` : '') }
