# HRC_taskAlloc_w.TB_w.AR

## 模型設計
### Baseline model
 - **分配方式**：分為Manual、Robot、HRC三種
 - **JOB Time**: 為合理的比較兩模型，JOB time使用動速執行時間及局部的最佳分配作為作業時間
### One-handed-task model
## 模型限制
 - **換手問題**：在組裝或拆解時完成時，應給予交由另一個代理搬運的選擇
 - **等待**：完成單手任務時應有在原地等待的選擇
 - **動作的連續性**：執行完任務A可以從當下的位置出發開始做任務B
 - **Binding限制**：只能處理一對一的binding關係
