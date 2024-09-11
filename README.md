# HRC_taskAlloc_w.TB_w.AR
## Introduction
* 在data資料夾中，可以修改
  * _data.xlsx：動素分析的結果，包含開始與結束位置及難易度
  * _oht_relation.csv：由tbHandler輸出的結果，取得各oht編號及其組成動素，判斷oht間的順序、聯結關係建立關聯性表格
  * _position.csv：定義座標點的位置(機械手臂坐標系；單位為cm)
  * _alloc_limit.csv：分配限制
* _bot_process_time.csv：需運行路徑模擬程式取得
* 執行main.py，依據固定格式輸入指令，決定要使用哪個模型進行最佳化求解，詳細指令可以輸入'-h'查看
## Model Design
### Data Structure (therbligsHandler.py)
* **層級大小**：Task -> OHT(one-handed task) -> Therblig
* **tb_abbr**：檢查動素名稱是否存在
* **Therblig**：記錄動素的名稱、開始位置、結束位置、難易度與總執行時間
* **OHT**：由Therbligs所組成
* **Task**：由OHT所組成，根據包含的OHT決定任務種類

### One-handed-task model (GAScheduling_oht.py)
* **分配方式**：分為左手、右手、機械手臂
* 主要的流程在run()，基本上就是GA流程
* **cal_makespan()**：計算適應度時，會考慮任務關聯性與代理干涉
* **revise_start_time()**：干涉判定後的修正

### Baseline model (GAScheduling_task.py)
* **分配方式**：分為Manual、Robot、HRC三種
* **Task Time**：為合理的比較兩模型，使用動素執行時間及局部的最佳分配作為作業時間
  
## Limitation
* **動作的連續性**：執行完任務A可以從當下的位置出發開始做任務B
* **Binding限制**：只能處理一對一的binding關係
* **機械手臂直角移動**
* **處理干涉的方式非最佳解**
  * 目前以第一次伸手所觸及的位置作為repr_pos