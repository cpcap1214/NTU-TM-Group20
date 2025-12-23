# Step 1：前處理

## 主要任務

- 先把每一個評論給標上正面 / 負面的標籤
    - 4, 5 星的平均是正面
    - 3, 2, 1 星的平均是負面

## 做了什麼

- 讀取原始評論資料（CSV），欄位為 place_name, rating, review_text
- 清理資料
    - rating 轉成數值，無法轉換或不在 1~5 範圍內的資料移除
    - place_name 或 review_text 為空值的資料移除
- 依照 rating 產生 sentiment 欄位（positive/negative）
- 輸出標註後的新 CSV（包含 place_name, rating, review_text, sentiment）
