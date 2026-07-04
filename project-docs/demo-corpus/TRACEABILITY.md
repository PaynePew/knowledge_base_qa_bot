# 文件 → F-ID 對照表（corpus v2, issue #440）

每對 zh/EN 鏡像文件敘述同一組事實條目。審核方式：抽一份文件，逐段對回 FACTS.md 的 F-ID。

| 主題 | zh-TW（docs/demo-zh/） | EN（docs/fake-docs/） | F-IDs |
|---|---|---|---|
| 關於 ACME | 關於ACME商店.md | acme_shop_about.md | F-ABT-01…03 |
| 聯絡客服 | 聯絡客服.md | contact_support.md | F-CON-01…03、F-GEN-03 |
| 帳號管理與安全 | 帳號管理與安全.md | account_management_security.md | F-ACC-01…05 |
| 付款方式 | 付款方式.md | payment_methods.md | F-PAY-01…06 |
| 國內配送 | 運送與配送.md | shipping_delivery.md | F-SHP-01…05 |
| 超商取貨 | 超商取貨.md | convenience_store_pickup.md | F-CVS-01…03、F-PAY-05（參照） |
| 門市自取 | 門市自取.md | store_pickup.md | F-PIC-01…04、F-ABT-02（參照） |
| 國際配送 | 國際配送.md | international_shipping.md | F-INT-01…05 |
| 訂單管理 | 訂單管理.md | order_management.md | F-ORD-01…03 |
| 退款與退貨 | 退款與退貨.md | refunds_returns.md | F-RET-01…07 |
| 損壞商品 | 損壞商品.md | damaged_items.md | F-DMG-01…03、F-RET-02（參照） |
| 商品保固 | 商品保固.md | warranty.md | F-WAR-01…04 |
| 會員與紅利點數 | 會員與紅利點數.md | membership_points.md | F-PTS-01…07 |
| 禮物卡 | 禮物卡.md | gift_cards.md | F-GIF-01…04、F-PRO-02（參照） |
| 促銷代碼 | 促銷代碼.md | promo_codes.md | F-PRO-01…03 |
| 訂閱訂單 | 訂閱訂單.md | subscription_orders.md | F-SUB-01…05 |

備註：

- EN entity 檔沿用 legacy basenames（`acme_shop_about.md`、`warranty.md`）——`CORPUS_ENTITY_SOURCES`（corpus_generator.py 與 sampling.py）及其測試把這兩個名字寫死，沿用可零改動。
- 幣別一律 NT$（F-GEN-01）；EN 檔不做美元換算。
- Gold Sections：概念文件共 62 節（≥50 門檻），entity 2 檔不計入。
