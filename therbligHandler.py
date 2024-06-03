#%% 
from math import ceil, nan
import numpy as np
import pandas as pd
import copy

#%% Test
tb_abbr = {
    "R": "Reach",
    "M": "Move",
    "G": "Grasp",
    "RL": "Release Load",
    "A": "Assemble",
    "DA": "Disassemble",
    "P": "Position",
    "H": "Hold"
}

AGENT = ["LH", "RH", "BOT"]


#%% 動素
class Therblig(object):
    def __init__(self, Name:str=None, From:str=None, To:str=None, Type:str=None, Obj1:str=None, Obj2:str=None):
        if tb_abbr.get(Name) == None:
            raise ValueError("This type of therblig is not exist")
        self.name = Name
        self.start: float
        self.end: float
        self.From = From
        self.To = To
        self.Type = Type # difficulty
        self.Obj1 = Obj1
        self.Obj2 = Obj2
        self.time = 0
        # self.next = None
        # self.tb_process_time = pd.read_excel("data_test.xlsx", sheet_name="Therblig Process Time")   
        
    def __repr__(self):
        return f"#{str(self.name)}"
    
    def get_tb_time(self, agent_pos, agent, POS, MTM):
        # print(f"MTM.loc[{self.type}, {AGENT[agent]}] * {np.lonalg.norm}")
        if self.name in ["R", "M"]:
            if self.From == "AGENT":
                dist = np.linalg.norm(POS[self.To] - POS[agent_pos[agent]])
            else:
                dist = np.linalg.norm(POS[self.To] - POS[self.From])
                
            if dist == 0:
                print(f"{self.To}, {self.From}")
                dist += 2
            elif dist <= 30:
                dist = ceil(dist / 2) * 2
            elif dist <= 80:
                dist = ceil(dist / 5) * 5
            else:
                dist = 80
            self.time = MTM.at[self.name + str(int(dist)) + self.Type, AGENT[agent]]
            return self.time
            # return MTM.at[self.type, AGENT[agent]] * np.linalg.norm(POS[self.To] - POS[self.From])
        else:
            self.time = MTM.at[self.name, AGENT[agent]]
            return MTM.at[self.name, AGENT[agent]]
        
    def is_moving_tb(self):
        if self.name in ["R", "M"]:
            return True
        return False


#%% 動素序列
class RawTherbligs(object):
    def __init__(self, Pos):
        self.list = []
        self.Pos = Pos
    def __repr__(self):
        return ", ".join(map(str, self.list))
        
    def read(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            therblig = Therblig(
                Name = row["Name"], 
                From = row["From"], 
                To = row["To"] if not pd.isna(row["To"]) else None, 
                Type = row["Type"],
                Obj1 = row["Obj1"], 
                Obj2 = row["Obj2"]
            )
            self.list.append(copy.copy(therblig))


#%% 單手任務
class OHT:
    def __init__(self, ls:list):
        self.id:int = -1
        self.tb_list:list = ls
        self.next:list = []
        self.prev:list = []
        self.bind:OHT = None
        self.To: str
        
        self.name = "PP"
        for tb in self.tb_list:
            if tb.name == "A":
                self.name = "A"
                break
            elif tb.name == "DA":
                self.name = "A"
                break
                
        # self.is_scheduled = False
            
    def __repr__(self):
        return "(" + ", ".join(map(str, self.tb_list)) + ")"
        # return f"OHT{self.id}"
    
    def set_id(self, id):
        self.id = id
    
    def add_next(self, n):
        self.next.append(n)
    
    def add_prev(self, p):
        self.prev.append(p)
    
    def get_oht_time(self, agent_pos, agent, POS, MTM):
        # print("????")
        oht_t = 0
        local_agent_pos = copy.copy(agent_pos)
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(local_agent_pos, agent, POS, MTM)
            if tb.name in ['R', 'M']:
                local_agent_pos[agent] = tb.To
        return oht_t
    
    def get_timestamp(self, agent_pos, agent, POS, MTM):
        oht_t = 0
        timestamps = []
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(agent_pos, agent, POS, MTM)
            if tb.is_moving_tb(): 
                timestamps.append((oht_t, POS[tb.To]))
        return timestamps
    
    def flat(self):
        return self.tb_list
    
    def renew_agent_pos(self, agent_pos, agent, pos):
        for tb in self.tb_list[::-1]:
            if tb.name in ['R', 'M']:
                agent_pos[agent] = tb.To
                break
        
#%% TBHandler
class TBHandler(object):
    def __init__(self, num_tbs, id):
        self.Pos = {}
        self.tbsl:RawTherbligs
        self.tbsr:RawTherbligs
        self.num_tbs = num_tbs
        self.id = id
        self.tbs_list = []
        self.job_list = []
        self.oht_list = []
    
    ## Save tbs by list
    def save_tbs(self):
        for i in range(1, self.num_tbs+1):
            tmp = RawTherbligs(self.Pos)
            tbs_df = pd.read_excel(f"data/data_{self.id}.xlsx", sheet_name=f"Therbligs{i}")
            tmp.read(tbs_df)
            self.tbs_list.append(tmp)
        
        # self.tbsl = RawTherbligs(self.Pos) # save pos
        # tbsl_df = pd.read_excel("data2.xlsx", sheet_name="Therbligs(L)")   
        # self.tbsl.read(tbsl_df)
        
        # self.tbsr = RawTherbligs(self.Pos)
        # tbsr_df = pd.read_excel("data2.xlsx", sheet_name="Therbligs(R)")   
        # self.tbsr.read(tbsr_df)
        
    ## Convert tbs to oht    
    def read_tbs(self):
        for tbs in self.tbs_list:
            if not isinstance(tbs, RawTherbligs):
                continue
            job = []
            tmp = []
            for tb in tbs.list:
                tmp.append(tb)
                if tb.name == "RL":
                    # self.list.append(tmp.copy())
                    oht = OHT(tmp.copy())
                    self.oht_list.append(oht)
                    job.append(oht)
                    tmp.clear()
            self.job_list.append(job)

        # Use dummy node to represent "END"
        self.oht_list.append(OHT([]))
                        
    def get_oht_time(self, oht_id, Pos):
        self.oht_list[oht_id].get_oht_time(Pos)
           
    def set_oht_id(self):
        for id, oht in enumerate(self.oht_list):
            oht.set_id(id)
     
    def run(self):
        self.save_tbs()
        self.read_tbs()
        self.set_oht_id()
        # print(self.OHT_list)



#%% Main
# myTBHandler = TBHandler()
# myTBHandler.run()