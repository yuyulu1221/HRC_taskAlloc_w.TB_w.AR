from enum import Enum
from math import ceil, nan
import numpy as np
import pandas as pd
import dataHandler as dh

## Check if the therblig name exists
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

class AgentType(Enum):
	LH = 0
	RH = 1
	BOT = 2

class Timestamp:
	def __init__(self, time, pos):
		self.time:int = time
		self.pos:str = pos

class Therblig:
    def __init__(self, Name:str=None, From:str=None, To:str=None, Type:str=None):
        if tb_abbr.get(Name) == None:
            raise ValueError(f"This type of therblig is not exist: {Name}")
        self.name = Name
        self.From = From
        self.To = To
        self.type = Type ## difficulty
        self.time = self.cal_tb_time()
        
    def __repr__(self):
        return f"#{str(self.name)}"    
    
    def cal_tb_time(self):
        ptime = [0, 0, 0]
        for ag in AgentType:
            if self.is_moving_tb():
                if ag == AgentType.BOT: # BOT
                    if self.From == "AGENT" and self.To == "AGENT":
                        print('Same position')
                    elif self.From =="AGENT":
                        p1, p2 = sorted([ag.name, self.To])
                    elif self.To == "AGENT":
                        p1, p2 = sorted([self.From, ag.name])
                    else:
                        p1, p2 = sorted([self.From, self.To])
                        
                    if p1 == p2:
                        ptime[ag.value] = 0
                    else:
                        ptime[ag.value] = int(dh.BOTM.at[f"{p1}<->{p2}", "Time"])
                else:
                    ## LH and RH: Read MTM table
                    if self.From == "AGENT":
                        dist = dh.cal_dist(dh.POS[self.To], dh.POS[ag.name])
                    elif self.To == "AGENT":
                        dist = dh.cal_dist(dh.POS[ag.name], dh.POS[self.From])
                    else:
                        dist = dh.cal_dist(dh.POS[self.To], dh.POS[self.From])
                    
                    if dist <= 30:
                        dist = ceil(dist / 2) * 2
                    elif dist <= 80:
                        dist = ceil(dist / 5) * 5
                    else:
                        dist = 80

                    ptime[ag.value] = int(dh.MTM.at[self.name + str(dist) + self.type, ag.name])         
            else:
                ptime[ag.value] = int(dh.MTM.at[self.name, ag.name])
                
        return ptime
    
    def get_tb_time(self, ag_pos:str, ag:AgentType) -> int:
        if self.From == 'AGENT':
            if ag == AgentType.BOT:
                p1, p2 = sorted([ag_pos, self.To])
                if p1 == p2:
                    return 0
                else:
                    return int(dh.BOTM.at[f"{p1}<->{p2}", "Time"])
            else:    
                dist = dh.cal_dist(dh.POS[self.To], dh.POS[ag.name])
                if dist <= 30:
                        dist = ceil(dist / 2) * 2
                elif dist <= 80:
                    dist = ceil(dist / 5) * 5
                else:
                    dist = 80
                return int(dh.MTM.at[self.name + str(dist) + self.type, ag.name])
        return self.time[ag.value]
    
    def is_moving_tb(self):
        return self.name in ["R", "M"]

#%% 單手任務
class OHT:
    def __init__(self, ls:list):
        self.id:int = -1
        self.tb_list:list = ls
        self.next:list = []
        self.prev:list = []
        self.bind:OHT = None
        self.bind_time:int = 0
        # self.end_time:int = 0
        self.To: str
        self.repr_pos:str = self.tb_list[0].To if len(self.tb_list) else ""
        self.type:str = self.decide_type()
        self.time = self.cal_oht_time()
        # self.prev_connect_time, self.next_connect_time = self.get_connect_time()
            
    def __repr__(self):
        return "(" + ", ".join(map(str, self.tb_list)) + ")"
    
    def decide_type(self) -> str:
        for tb in self.tb_list:
            if tb.name == "A":
                return "A"
            elif tb.name == "DA":
                return "DA"
        return "P&P"
        
    def cal_oht_time(self):
        ptime = [0, 0, 0]
        for ag in AgentType:
            for tb in self.tb_list:
                ptime[ag.value] += tb.get_tb_time(ag.name, ag)
        return ptime
    
    ## Try to address the issue of the path difference between the human and the robotic arm
    # def get_connect_time(self):
    #     prev_connect_time = [0, 0, 0]
    #     next_connect_time = [0, 0, 0]
    #     for ag in AgentType:
    #         cur_total = 0
    #         for i, tb in enumerate(self.tb_list):
    #             cur_total += tb.get_tb_time(ag.name, ag)
    #             if tb.name == 'G':
    #                 prev_connect_time[ag.value] = cur_total
    #         cur_total = 0
    #         for i, tb in enumerate(self.tb_list[::-1]):
    #             if tb.name == 'A' and self.tb_list[i+1].name == 'RL':
    #                 next_connect_time[ag.value] = cur_total
    #                 break
    #             if tb.name == 'RL'and self.tb_list[i-1].name != 'A':
    #                 next_connect_time[ag.value] = cur_total
    #                 break
    #             cur_total += tb.get_tb_time(ag.name, ag)
    #     return prev_connect_time, next_connect_time
                    
    
    def get_oht_time(self, ag_pos, ag):
        return self.time[ag.value]
    
    ## For AR system
    def get_process_method(self, ag) -> dict:
        data = []
        for tb in self.tb_list:
            data.append(dict(
                TaskId = self.id,
                Name = tb.name,
                From = tb.From if tb.From != 'AGENT' else ag.name,
                To = tb.To if tb.To != 'AGENT' else ag.name,
                time = tb.time[ag.value]
            ))
        return data
    
    def get_bind_remain_time(self, ag_pos, ag):
        rem_t = 0
        for tb in self.tb_list[::-1]:
            if tb.name in ['A', 'DA']:
                break
            rem_t += tb.get_tb_time(ag_pos, ag)
        return rem_t
    
    def get_timestamp(self, ag_pos, ag):
        oht_t = 0
        timestamps = []
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(ag_pos, ag)
            if tb.is_moving_tb(): 
                if tb.To == 'AGENT':
                    timestamps.append(Timestamp(oht_t, ag_pos))
                else: 
                    timestamps.append(Timestamp(oht_t, tb.To))
        return timestamps
    
    def flat(self):
        return self.tb_list
    
    def renew_agent_pos(self, ag_pos_d, ag_id):
        if ag_id != 2:
            return
        # ## Find the last movable therblig
        for tb in self.tb_list[-2::-1]:
            if tb.is_moving_tb():
                ag_pos_d[ag_id] = tb.To
                break
        
            
class TASK:
    def __init__(self, ls:list):
        self.oht_list = ls
        self.type = "P&P"
        for oht in self.oht_list:
            if oht.type == "A":
                self.type = "A"
                break
            elif oht.type == "DA":
                self.type = "DA"
                break
    def flat(self):
        return self.oht_list
        
        
#%% TBHandler
class TBHandler(object):
    def __init__(self, num_tbs, id):
        self.Pos = {}
        self.num_tbs = num_tbs
        self.id = id
        self.task_list = []
        self.oht_list = []
    
    ## Save tbs by list
    def save_tbs(self):
        for i in range(1, self.num_tbs+1):
            tbs_df = pd.read_excel(f"data/{self.id}_data.xlsx", sheet_name=f"Therbligs{i}")
            tmp = []
            task = []
            for _, row in tbs_df.iterrows():
                if row['Name'] == 'END':
                    oht = OHT(tmp.copy())
                    self.oht_list.append(oht)
                    task.append(oht)
                    tmp.clear() 
                    continue
                
                therblig = Therblig(
                    Name = row["Name"], 
                    From = row["From"] if not pd.isna(row["To"]) else None, 
                    To = row["To"] if not pd.isna(row["To"]) else None, 
                    Type = row["Type"] if not pd.isna(row["To"]) else None, 
                )
                tmp.append(therblig)
            self.task_list.append(TASK(task))
            
        ## Use dummy node to represent "END" OHT   
        self.oht_list.append(OHT([]))                   
           
    def set_oht_id(self):
        for id, oht in enumerate(self.oht_list):
            oht.id = id
            
    def write_process_method(self, ag):
        pm = []
        for oht in self.oht_list:
            pm.extend(oht.get_process_method(ag))
            
        pm_df = pd.DataFrame(pm)
        pm_df.to_csv(f"./result/{self.id}_process_method_{ag.name}.csv" ,index=False)
     
    def run(self):
        self.save_tbs()
        self.set_oht_id()
        
        ## Output only the method of the robotic arm
        self.write_process_method(AgentType.BOT)