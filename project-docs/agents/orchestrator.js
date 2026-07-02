export const meta = {
  name: 'kb-slice-orchestrator',
  description: '專案自有 orchestrator（gh-native，零 beads）：plan(GitHub issues / 明確 slices) → 每 issue 並行 build(implement+review 同一 worktree) → 對抗驗證(嚴重度閘門) → 過閘門備妥分支（openPRs:true 才另跑 merge agent 開 PR；人類永遠負責按下 merge）',
  phases: [{ title: 'Plan' }, { title: 'Build' }, { title: 'OpenPRs' }],
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
  openPRs     : false,      // 預設只備妥+驗證分支，回報 eligible，由人類（頂層 session）開 PR + merge。true 才另跑 merge agent 開 PR（人類仍負責按 merge）
  slices      : null,       // 明確切片清單 [{id,title,branch?}]；給了就用它當來源，完全不碰任何 tracker
  only        : null,       // 只做這些 id（gh 來源時當過濾；配 skipPlan 時當明確來源）
  skipPlan    : false,      // 配 only：跳過 gh 查詢，直接 build
  repo        : 'PaynePew/knowledge_base_qa_bot',
  ..._argsObj,
}
const MODELS = { plan:'haiku', build:'sonnet', verify:'opus', merge:'haiku', note:'haiku', ...(_argsObj.models ? _argsObj.models : {}) }

log(`effective config → slices=${CONFIG.slices ? CONFIG.slices.length : 'null'} · only=${JSON.stringify(CONFIG.only)} · adversarial=${CONFIG.adversarial} · openPRs=${CONFIG.openPRs} · base=${CONFIG.baseBranch} · standards=${CONFIG.standards}`)

const PLAN_SCHEMA = { type:'object', required:['issues'], properties:{
  issues:{ type:'array', items:{ type:'object', required:['id','branch'], properties:{
    id:{type:'string'}, title:{type:'string'}, type:{type:'string'}, branch:{type:'string'} } } } } }
const VERDICT_SCHEMA = { type:'object', required:['verdict','blockers'], properties:{
  verdict:{ type:'string', enum:['pass','changes-requested'] },
  blockers:{ type:'array', items:{ type:'object', required:['severity'], properties:{
    file:{type:'string'}, line:{type:'number'}, issue:{type:'string'},
    severity:{ type:'string', enum:['critical','high','medium','low'] } } } } } }

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
const verifyPrompt = (i) => `唯讀對抗式驗證 GitHub issue #${i.id}：執行 git diff ${CONFIG.baseBranch}..${i.branch}，盡力找正確性與安全 bug，並對照 ${CONFIG.standards}（特別是 §11 drift signals）與相關 Accepted project-docs/adr/* 檢查違規。
若 diff 顯示它刪除/還原了既有成果（其他 slice、ADR、基礎設施），或新增了 .beads/.agents/AGENTS.md / 動了 CLAUDE.md / .claude 設定，標 critical（多半代表 base 拿錯或 agent 越界）。
每個 blocker 標 severity：critical/high＝會壞/不安全/刪到別人成果/越界污染（擋 merge）；medium/low＝小毛病（回報不擋）。
不要改 code、不要 commit。回 verdict（pass＝無 critical/high；否則 changes-requested）與 blockers（file,line,issue,severity）。`

// note：把 implementer 摘要 + reviewer 結論貼成 GitHub issue 留言（gh only；不改 code、不 commit）。預設不跑（openPRs 才跑），最小化 agent 動作面。
const commentPrompt = (i, r) => `你的唯一任務：把 slice #${i.id} 的階段成果貼成該 GitHub issue 的留言，寫完即止。
只執行一筆：gh issue comment ${i.id} --repo ${CONFIG.repo} --body-file -（內容從 stdin 餵入，避免 shell 解析）。
${GUARDRAILS}
留言內容：
=== IMPLEMENTER ===
${String(r?.build ?? '(無摘要)').slice(0, 2000)}
=== /IMPLEMENTER ===
=== REVIEWER ===
verdict=${r?.v?.verdict ?? 'n/a'}；擋 merge 的 blocker ${(r?.blocking ?? []).length} 個。
${JSON.stringify(r?.blocking ?? [], null, 2)}
=== /REVIEWER ===
只貼這一筆，不要做其他事。`

// merge（openPRs:true 才跑）：讀專案 merge.md，push 分支 + 開 PR（Closes #N）；人類負責按 merge
const mergePrompt = (i) => `讀 ${CONFIG.promptsDir}/merge.md 照做：把分支 push 到 origin 並開 PR。代入 {{ISSUE}}=${i.id}、{{BRANCH}}=${i.branch}、{{TARGET_BRANCH}}=${CONFIG.baseBranch}。
${GUARDRAILS}
注意：只 push + 開 PR（body 含 Closes #${i.id}）+ 在 issue 留 PR 連結；絕不自己按 merge、絕不 gh issue close（GitHub 會在人類 merge PR 時自動關）。
回傳：PR 連結與狀態。`

// ── ① PLAN：明確 slices / only（零 tracker）優先；否則 gh issue list（GitHub 原生佇列）──
let todo
if (CONFIG.slices?.length) {
  todo = CONFIG.slices.map(s => ({ id:String(s.id), title:s.title || '', type:s.type || 'task', branch: s.branch || mkBranch(s.id) }))
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
           note: `分支已備妥並過閘門（從 ${CONFIG.baseBranch} 開）。由頂層 session 逐一：驗證真實 artifact → 開 PR(Closes #N) → CI 雙綠 → 人工 merge。或傳 {openPRs:true} 讓 merge agent 代開 PR（人類仍負責按 merge）。` }

// ── ③ OPEN PRs（openPRs:true）：每 eligible 分支 push + 開 PR（Closes #N）；人類負責 merge ──
phase('OpenPRs')
const prs = await parallel(eligible.map(e => () =>
  agent(mergePrompt(e.issue), { label:`openpr:${lbl(e.issue)}`, phase:'OpenPRs', model: MODELS.merge })
    .then(out => ({ id: e.issue.id, out })).catch(() => ({ id: e.issue.id, out: null }))))
return { planned: todo.length, opened: prs.filter(p => p.out).length, blocked, prs,
         note: 'PR 已開（Closes #N）。CI 雙綠 + 你確認無誤後由人類 merge；orchestrator 不自動 merge。' }
