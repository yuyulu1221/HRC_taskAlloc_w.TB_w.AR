#%% 
from math import ceil, nan
from os import TMP_MAX
import numpy as np
import pandas as pd
import copy
import dataHandler as dh

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
    def __init__(self, Name:str=None, From:str=None, To:str=None, Type:str=None):
        if tb_abbr.get(Name) == None:
            raise ValueError("This type of therblig is not exist")
        self.name = Name
        self.start: float
        self.end: float
        self.From = From
        self.To = To
        self.type = Type # difficulty
        
    def __repr__(self):
        return f"#{str(self.name)}"    
    
    def get_tb_time(self, ag_pos, ag_id) -> int:
        # print(f"MTM.loc[{self.type}, {AGENT[agent]}] * {np.lonalg.norm}")
        if self.name in ["R", "M"]:
            if ag_id == 2: # BOT
                if self.From =="AGENT":
                    p1 = ag_pos
                    p1, p2 = sorted([p1, self.To])
                else:
                    p1, p2 = sorted([self.From, self.To])
                self.time = dh.BOTM.at[f"{p1}<->{p2}", "Time"]
                return int(self.time)

            if self.From == "AGENT":
                dist = np.linalg.norm(dh.POS[self.To] - dh.POS[ag_pos])
            else:
                dist = np.linalg.norm(dh.POS[self.To] - dh.POS[self.From])
            
            if dist <= 30:
                dist = ceil(dist / 2) * 2
            elif dist <= 80:
                dist = ceil(dist / 5) * 5
            else:
                dist = 80
            self.time = dh.MTM.at[self.name + str(dist) + self.type, AGENT[ag_id]]
            return int(self.time)
        
        else:
            self.time = dh.MTM.at[self.name, AGENT[ag_id]]
            return dh.MTM.at[self.name, AGENT[ag_id]]
        
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
        self.bind_time = 0
        self.To: str
        self.type = "P&P"
        for tb in self.tb_list:
            if tb.name == "A":
                self.type = "A"
                break
            elif tb.name == "DA":
                self.type = "A"
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
    
    def get_oht_time(self, ag_pos, ag_id):
        oht_t = 0
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(ag_pos, ag_id)
        return oht_t
    
    def get_bind_remain_time(self, ag_pos, ag_id):
        rem_t = 0
        for tb in self.tb_list[::-1]:
            if tb.name in ['A', 'DA']:
                break
            rem_t += tb.get_tb_time(ag_pos, ag_id)
        return rem_t
    
    def get_timestamp(self, ag_pos, ag_id):
        oht_t = 0
        timestamps = []
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(ag_pos, ag_id)
            if tb.is_moving_tb(): 
                timestamps.append((oht_t, dh.POS[tb.To]))
        return timestamps
    
    def flat(self):
        return self.tb_list
    
    def renew_agent_pos(self, agent_pos, agent):
        for tb in self.tb_list[::-1]:
            if tb.name in ['R', 'M']:
                agent_pos[agent] = tb.To
                break
            
class JOB:
    def __init__(self, ls:list):
        self.oht_list = ls
        self.type = "P&P"
        for oht in self.oht_list:
            if oht.type == "A":
                self.type = "A"
                break
            elif oht.type == "DA":
                self.type = "A"
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
            self.job_list.append(JOB(job))

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