# ACME 商店 — Demo 語料唯一事實來源（FACTS.md）

> **狀態：APPROVED 2026-07-04**（issue #440 HITL 關卡 1 已過）。
> 修訂 2026-07-04（bake 迭代）：F-PAY-01、F-CVS-03、F-PIC-03、F-DMG-01、F-DMG-03 依 grounding 驗證揭露的資訊縫隙補明確化細節——最終審核時請一併過目。
> 修訂 2026-07-05（bake 迭代 2）：F-PAY-04 補「自下單起」（gpt-5 clean bake 12 頁隔離的唯一真事實縫隙；其餘 11 處為源文件引言句/措辭補強，未動事實）。
> 規則：demo 語料（zh-TW 與 EN 鏡像）裡出現的**每一個數字與規則**都必須能回溯到本表的一個事實條目（`F-*` ID）。
> 本表沒有的事實 = 語料不准寫。要新增事實，先改本表再改語料。
> 刻意種植的瑕疵（lint demo 用）記錄在文末 §P，與正常事實嚴格分開。

## 幣別與市場

- **F-GEN-01** 幣別一律新台幣（NT$ / 新台幣）。EN 鏡像文件同樣使用 NT$，不做美元換算。
- **F-GEN-02** 市場：台灣（本島 + 離島澎湖、金門、馬祖）；國際配送見 §INT。
- **F-GEN-03** 工作天定義：週一至週五，不含國定假日。

## §ABOUT 關於 ACME 商店

- **F-ABT-01** ACME 商店創立於 2015 年，總部位於台北，是主打居家生活用品與 3C 配件的線上購物平台。
- **F-ABT-02** 實體門市共 3 家：台北信義、台中西屯、高雄夢時代。
- **F-ABT-03** 自有品牌「ACME Basics」約佔商品數三成，其餘為代理品牌。

## §CONTACT 聯絡客服

- **F-CON-01** 客服管道：線上客服（官網聊天視窗）、Email（support@acme-shop.example）、客服電話 02-1234-5678。
- **F-CON-02** 客服時間：週一至週五 09:00–18:00；線上客服另提供週六 10:00–17:00。
- **F-CON-03** Email 回覆 SLA：1 個工作天內。

## §ACCT 帳號管理與安全

- **F-ACC-01** 註冊方式：Email、Google、LINE 三種。
- **F-ACC-02** 密碼規則：至少 10 碼，需含大小寫字母與數字。
- **F-ACC-03** 支援兩步驟驗證（簡訊 OTP 或驗證器 App），於「帳號設定 → 安全性」開啟。
- **F-ACC-04** 連續 5 次登入失敗鎖定帳號 30 分鐘。
- **F-ACC-05** 帳號刪除：於「帳號設定」提出申請後 14 天猶豫期，期滿永久刪除；期間可登入取消。

## §PAY 付款方式

- **F-PAY-01** 信用卡：VISA、MasterCard、JCB；可一次付清，滿額訂單亦可分期（見 F-PAY-02）。
- **F-PAY-02** 信用卡分期：3、6、12 期；單筆滿 NT$3,000 才可分期；3/6 期零利率，12 期年利率 5%。
- **F-PAY-03** 行動支付：Apple Pay、Google Pay。
- **F-PAY-04** 超商代碼繳費：單筆上限 NT$20,000，繳費期限自下單起 3 天，逾期訂單自動取消。
- **F-PAY-05** 貨到付款：僅限本島超商取貨，單筆上限 NT$20,000，另收手續費 NT$30。
- **F-PAY-06** 不接受：銀行轉帳、支票、外幣付款。

## §SHIP 運送與配送（國內）

- **F-SHP-01** 標準配送：本島出貨後 3–5 個工作天送達；離島另加 1–3 個工作天。
- **F-SHP-02** 標準配送運費 NT$80；單筆訂單滿 NT$1,000 免運。
- **F-SHP-03** 快速到貨：本島出貨後 1–2 個工作天，運費 NT$150，需於當日 14:00 前完成下單付款；大型/易碎商品不適用。
- **F-SHP-04** 出貨時間：付款確認後 24–48 小時內出貨；大型促銷檔期可能延後 1–2 天。
- **F-SHP-05** 物流追蹤：出貨後寄送含物流單號的通知信，物流資訊於出貨後 24 小時內更新。

## §CVS 超商取貨

- **F-CVS-01** 運費 NT$60（不計入 F-SHP-02 的免運門檻優惠——滿 NT$1,000 超商取貨同樣免運）。
- **F-CVS-02** 到店後保留 7 天，逾期退回並自動取消訂單（款項依原付款方式退回）。
- **F-CVS-03** 限制：單件超過 5 公斤，或任一邊長超過 45 公分（材積過大）不適用。

## §PICKUP 門市自取

- **F-PIC-01** 三家門市（F-ABT-02）皆提供自取，免運費。
- **F-PIC-02** 訂單狀態顯示「可取貨」後才可前往，一般為付款後 2 小時內（門市營業時間內）。
- **F-PIC-03** 自「可取貨」通知起保留 3 天（日曆天），逾期自動取消並全額退款，依 F-RET-04 時程退回原付款方式。
- **F-PIC-04** 取貨需出示訂單條碼與身分證件。

## §INT 國際配送

- **F-INT-01** 配送地區：日本、韓國、香港、澳門、新加坡、馬來西亞、美國、加拿大。
- **F-INT-02** 時效：亞洲 5–10 個工作天；北美 10–15 個工作天。
- **F-INT-03** 運費：亞洲 NT$400 起、北美 NT$800 起，依重量於結帳時計算；國際訂單不適用免運。
- **F-INT-04** 關稅與進口稅由收件人負擔（DDU）。
- **F-INT-05** 排除品項：液體、電池類商品不提供國際配送。

## §ORDER 訂單管理

- **F-ORD-01** 出貨前（狀態為「處理中」）可取消訂單或修改收件地址；出貨後不可取消，需改走退貨（§RET）。
- **F-ORD-02** 下單後 1 小時內可自助修改付款方式；逾時需取消重下。
- **F-ORD-03** 訂單紀錄保留 5 年，可於「我的訂單」查詢與下載發票。

## §RET 退款與退貨

- **F-RET-01** 退款申請窗口：收到商品後 14 天內（含例假日）。
- **F-RET-02** 條件：商品未使用、包裝完整、附原始購買憑證；瑕疵或運送損壞不受此限。
- **F-RET-03** 審核時間：客服 3 個工作天內審核。
- **F-RET-04** 退款時效：審核通過後 5–7 個工作天退回原付款方式。
- **F-RET-05** 退貨運費：非瑕疵退貨由顧客負擔（NT$80，自退款扣除）；瑕疵退貨由 ACME 負擔。
- **F-RET-06** 不適用退款：已下載的數位內容、客製化商品（刻字/印花）、已開封的食品與衛生用品。
- **F-RET-07** 無任何品類收取整新費（restocking fee）。

## §DMG 損壞商品

- **F-DMG-01** 收到損壞/瑕疵商品：48 小時內透過「我的訂單 → 回報問題」上傳照片（商品與外包裝，需清楚顯示損壞部位）。
- **F-DMG-02** 處理選項：免費換貨或全額退款（含來回運費）。
- **F-DMG-03** 換貨出貨時效：審核通過後 3 個工作天內寄出（以出貨時間計，非送達時間）。

## §WAR 商品保固

- **F-WAR-01** 3C 配件與電器類：原廠保固 12 個月，自發票日起算。
- **F-WAR-02** ACME Basics 自有品牌：保固 24 個月。
- **F-WAR-03** 保固申請：官網「保固服務」填單 + 發票或訂單編號；維修期間 7–14 個工作天。
- **F-WAR-04** 人為損壞、超過保固期不在保固範圍；可付費維修，報價後顧客確認。

## §PTS 會員與紅利點數

- **F-PTS-01** 累積：每消費 NT$100 得 1 點；以實際付款金額計，運費與點數折抵部分不累積。
- **F-PTS-02** 入帳：訂單完成（鑑賞期過後）24 小時內。
- **F-PTS-03** 折抵：1 點 = NT$1，單筆最多折抵訂單金額 50%。
- **F-PTS-04** 效期：入帳日起 12 個月，逾期自動失效。
- **F-PTS-05** 推薦好友：好友完成首購後，推薦人得 50 點。
- **F-PTS-06** 點數不得兌換現金、不得轉讓；退款時已折抵/已回饋點數一併收回。
- **F-PTS-07** 會員等級：一般會員（註冊即是）→ 金卡（近 12 個月消費滿 NT$30,000：免運不限金額 + 生日當月雙倍點數）→ 白金（滿 NT$100,000：金卡權益 + 免費升級快速到貨 + 專屬客服線）。

## §GIFT 禮物卡

- **F-GIF-01** 面額：NT$500、NT$1,000、NT$3,000；電子形式，Email 交付。
- **F-GIF-02** 效期：無使用期限（依台灣禮券規範）。
- **F-GIF-03** 購買禮物卡不累積紅利點數（呼應 F-PTS-01 的「實際付款金額」）；用禮物卡付款的訂單正常累積。
- **F-GIF-04** 禮物卡不可退款、不可兌現、不可儲值找零；餘額保留於帳戶供下次使用。

## §PROMO 促銷代碼

- **F-PRO-01** 每筆訂單限用 1 組促銷代碼；可與紅利點數折抵並用（先折代碼再折點數）。
- **F-PRO-02** 促銷代碼不適用於禮物卡與已在特價中的商品。
- **F-PRO-03** 代碼逾期不補發；結帳時輸入，事後不可追加。

## §SUB 訂閱訂單

- **F-SUB-01** 適用品項：消耗性商品（如濾芯、清潔用品）；週期可選 30 / 60 / 90 天。
- **F-SUB-02** 訂閱價：每期享 95 折，並免運（不受 F-SHP-02 門檻限制）。
- **F-SUB-03** 暫停：最多可暫停 3 個月，需於下一個帳單週期至少 24 小時前操作。
- **F-SUB-04** 取消：需於下一個帳單週期至少 24 小時前完成；已排定出貨的該期仍會如期出貨，除非聯絡客服申請按比例退款。
- **F-SUB-05** 重新啟用已取消的訂閱：以當時價格與供貨狀況為準。

---

## 語料規格

- **主體**：上述 16 主題 × 2 語言（zh-TW + EN 鏡像）≈ 32 份文件；zh 檔名用中文、EN 檔名用英文 slug。
- **寫作紀律**：每份 3–6 個 `##` 小節；不寫行銷填充語；每個數字/規則後不標 F-ID（顧客看不到內部編號），但 PR 描述需附「文件 → F-ID 對照表」供審核。
- **鏡像一致性**：zh/EN 同主題文件敘述同一組 F-ID，允許語言慣用差異，不允許事實差異。

## §P 種植瑕疵（lint demo 劇本 — 2026-07-05 種植）

> 每類 lint 檢查 2 個、雙語分布；每一筆記錄：類別（C1–C12）、載體、瑕疵內容、預期修復動作。
> 種植的矛盾**只准**與本表衝突或與另一份種植文件衝突，不准改動乾淨語料。
> 種植載體 = 10 份新增源文件（實體隔離於 docs/planted-zh/ 與 docs/planted-en/，不進 Phase-8 eval 池 docs/fake-docs/——corpus_generator 的 DOC_SPECS 覆蓋測試因此不受種植影響；C11 載體兩檔已依劇本刪除）（單一小節，刻意精簡於語料規格的 3–6 節；不列入 TRACEABILITY 主表）＋烤製後的 wiki 層操作。除本表登記的瑕疵外，種植文件的其餘敘述一律回溯既有 F-ID。

**種植源文件（10 份）：**

| 文件 | 語言 | 承載 | 乾淨部分回溯 |
|---|---|---|---|
| planted-zh/退貨期限提醒.md | zh | C5-zh、C2-zh | F-RET-03/04/05 |
| planted-en/returns_reminder.md | en | C5-en、C2-en | F-RET-03/04/05 |
| demo-zh/快閃特賣.md | zh | C11-zh | F-PRO-01/02、F-SHP-02/04 |
| fake-docs/flash_sale_faq.md | en | C11-en | F-PRO-01/02、F-SHP-02/04 |
| planted-zh/門市服務指南.md | zh | C6-zh、C12-zh | F-ABT-02、F-PIC-01/04 |
| planted-en/store_services_guide.md | en | C6-en、C12-en(1/2) | F-ABT-02、F-PIC-01/04 |
| planted-zh/取貨方式比較.md | zh | C3-zh | F-SHP-02、F-CVS-01/02、F-PIC-01/03 |
| planted-en/pickup_options.md | en | C3-en | F-SHP-02、F-CVS-01/02、F-PIC-01/03 |
| planted-zh/會員日活動.md | zh | C4-zh | F-PTS-01/03/07 |
| planted-en/member_day.md | en | C4-en、C12-en(2/2) | F-PTS-01/03/07 |

**逐項登記（11 類 × 2，共 22 筆）：**

| # | 類別 | 語言 | 載體 | 瑕疵 | 預期修復 |
|---|---|---|---|---|---|
| P-01 | C5 direct | zh | 退貨期限提醒（頁）vs 退款申請窗口（頁） | 直述「收到商品後 30 天內」抵觸 F-RET-01（14 天）——初版連假框架被 judge 視為政策延伸不判矛盾，故改鈍化直述 | Reconcile |
| P-02 | C5 direct | en | return-deadline-reminder（頁）vs refund-window（頁） | 直述 "within 30 days" 抵觸 F-RET-01 | Reconcile |
| P-03 | C6 stale | zh | 門市服務指南.md（源） | 烤後源文件補上 F-PIC-02/03 細節 → docs_body hash 漂移 | Re-ingest |
| P-04 | C6 stale | en | store_services_guide.md（源） | 同 P-03（EN 鏡像） | Re-ingest |
| P-05 | C3 failed-grounding | zh | 取貨方式比較（頁） | frontmatter 翻成 `status: failed_grounding` + `verifier_unavailable`（模擬烤製時 verifier 斷線） | Re-ingest (retry) |
| P-06 | C3 failed-grounding | en | pickup-options-compared（頁） | body 手植無佐證句「Curbside pickup is available at all three stores.」+ `claim_unsupported` 列該句 | Fix Source（或 force Re-ingest 重合成） |
| P-07 | C11 full orphan | zh | 快閃特賣（頁） | 烤後刪除源文件 docs/demo-zh/快閃特賣.md | Confirmed delete |
| P-08 | C11 full orphan | en | flash-sale（頁） | 烤後刪除源文件 fake-docs/flash_sale_faq.md | Confirmed delete |
| P-09 | C4 collision | zh | 會員日活動.md（源） | 同一文件兩個 `## 會員日優惠` 小節 → 頁 + 頁-2 | Merge / Differentiate |
| P-10 | C4 collision | en | member_day.md（源） | 兩個 `## Member Day perks` 小節 | Merge / Differentiate |
| P-11 | C12 alias_vs_slug | zh | 門市服務指南（頁） | `aliases:` 加上某乾淨頁的既有 slug（烤後定值） | Assign Alias 重指派 |
| P-12 | C12 alias_vs_alias | en | store-services + member-day 兩頁 | 兩頁同時宣告 `aliases: [gift-vouchers]`（無此真頁） | Assign Alias |
| P-13 | C2 red link | zh | 退貨期限提醒（頁 body） | 附加「詳見 [[退貨教學]]」（無此頁） | Fill via Import |
| P-14 | C2 red link | en | return-deadline-reminder（頁 body） | 附加 "See also [[return-shipping-guide]]" | Fill via Import |
| P-15 | C1 coverage gap | zh | wiki/log.md | 3 行 `chat_fallback`「ACME 有提供禮物包裝服務嗎？」reason=retrieval_empty（FACTS 刻意無此事實） | Fill via Import → Verify re-ask |
| P-16 | C1 coverage gap | en | wiki/log.md | 2 行 `chat_fallback` "does ACME offer gift wrapping" reason=retrieval_empty | Fill via Import → Verify re-ask |
| P-17 | C8 promotion | zh | wiki/qa/（手寫 draft） | 會員日問題 draft、count 3 | Promote / Discard |
| P-18 | C8 promotion | en | wiki/qa/（手寫 draft） | flash-sale 問題 draft、count 1 | Promote / Discard |
| P-19 | C9 stale-qa | zh | wiki/qa/（手寫 live） | live qa 引用門市服務指南頁，qa.updated 早於該頁 updated | Re-file |
| P-20 | C9 stale-qa | en | wiki/qa/（手寫 live） | live qa 引用 store-services 頁，updated 較舊 | Re-file |
| P-21 | C10 invalid schema | en | wiki/qa/（手寫） | `status: Live`（大小寫錯誤） | 修 frontmatter 或 Discard |
| P-22 | C10 invalid schema | zh | wiki/qa/（手寫） | `count: 0`（非正整數） | 修 frontmatter 或 Discard |

C5 特別政策（LLM-judged check 的務實界線）：C5 由 LLM judge 判定、天生機率性，種植後的期望集合＝上表 2 筆設計 direct 對＋一份「凍結殘量」清單（種植完成時以實際 lint 輸出列舉並凍結於下方；多屬 judge 對相關政策的 tension 判讀）。凍結清單以外的新增 C5 才算污染。此外本輪已做的結構性降噪：root 測試 fixture 三檔的頁面退出 wiki 層（docs 保留給測試與 RAG）；超商取貨/門市自取四份源文件加入互斥對照措辭，消除 judge 混淆兩種保留期的 direct 誤判。

**C5 凍結清單（2026-07-05 種植完成時的實際 lint 輸出，共 22 對；委員 = 運行時預設 mini、verdict cache 隨 seed 提交，故 prod 首次 deep-audit 會確定性重現本清單）：**

| severity | page_a | page_b | 備註 |
|---|---|---|---|
| direct | 搭配貨到付款 | 貨到付款 |  |
| tension | cancelling | pausing |  |
| tension | convenience-store-pickup | item-restrictions |  |
| tension | fee | standard-home-delivery |  |
| tension | gift-cards | validity |  |
| tension | hold-period | pickup-flow-deadline |  |
| tension | international-shipping | shipping-delivery |  |
| tension | process | return-deadline-reminder | （設計載體） |
| tension | 取消訂閱 | 暫停訂閱 |  |
| tension | 退款流程 | 退貨期限提醒 | （設計載體） |
| duplicate | acme-shop-about | store-locations-and-pickup-service |  |
| duplicate | conditions | difference-from-regular-returns |  |
| duplicate | difference-from-regular-returns | return-shipping |  |
| duplicate | exclusions | gift-card-restrictions |  |
| duplicate | fee | pickup-options-compared |  |
| duplicate | locations | store-locations-and-pickup-service |  |
| duplicate | member-day-perks-2 | redeeming-points |  |
| duplicate | pickup-flow-deadline | pickup-options-compared |  |
| duplicate | 保留期限 | 取件流程與期限 |  |
| duplicate | 取貨方式比較 | 超商取貨運費 |  |
| duplicate | 會員日優惠-2 | 點數折抵 |  |
| duplicate | 自取地點 | 門市據點與自取服務 |  |

設計的 P-01/P-02（30 天 vs 14 天）由「退貨期限提醒 / return-deadline-reminder」載體頁承載，被 judge 配對到退款流程頁並判 tension——衝突有被抓到、Reconcile 修復動線成立；其餘為 judge 對相關政策頁的 tension/duplicate 判讀殘量，屬 C5 檢查本身的機率性行為，非語料錯誤。凍結清單以外的新增 C5 才算污染。

種植不變式：eval/lint_fixtures/ 完全不動（測試套件自足）；乾淨語料 32+3 檔零改動；C6/C9 靠 hash 與 frontmatter `updated`（loader 的 mtime-touch 是死碼，不依賴）；期望 findings 清單種植後以 `POST /wiki/lint?include_c5=true` 對照驗證 — 多一筆（organic noise）或少一筆都算種植失敗，回頭修種植文件。
