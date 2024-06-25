from enum import Enum
from math import ceil, nan
import numpy as np
import pandas as pd
import dataHandler as dh

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

class Timestamp:
	def __init__(self, time, pos):
		self.time:int = time
		self.pos:str = pos

class Therblig:
    def __init__(self, Name:str=None, From:str=None, To:str=None, Type:str=None):
        if tb_abbr.get(Name) == None:
            raise ValueError(f"This type of therblig is not exist: {Name}")
        self.name = Name
        self.start: float
        self.end: float
        self.From = From
        self.To = To
        self.type = Type # difficulty
        self.time = self.cal_tb_time()
        
    def __repr__(self):
        return f"#{str(self.name)}"    
    
    def cal_tb_time(self):
        ptime = [0, 0, 0]
        for ag_id in range(3):
            if self.is_moving_tb():
                # print(AGENT[ag_id], ag_pos, self.From, self.To)
                if ag_id == 2: # BOT
                    if self.From == "AGENT" and self.To == "AGENT":
                        print('Same position??')
                    elif self.From =="AGENT":
                        p1, p2 = sorted([AGENT[ag_id], self.To])
                    elif self.To == "AGENT":
                        p1, p2 = sorted([self.From, AGENT[ag_id]])
                    else:
                        p1, p2 = sorted([self.From, self.To])
                        
                    if p1 == p2:
                        ptime[ag_id] = 0
                    else:
                        ptime[ag_id] = int(dh.BOTM.at[f"{p1}<->{p2}", "Time"])
                else:
                    ## LH and RH: Read MTM table
                    if self.From == "AGENT":
                        dist = dh.cal_dist(dh.POS[self.To], dh.POS[AGENT[ag_id]])
                    elif self.To == "AGENT":
                        dist = dh.cal_dist(dh.POS[AGENT[ag_id]], dh.POS[self.From])
                    else:
                        dist = dh.cal_dist(dh.POS[self.To], dh.POS[self.From])
                    
                    if dist <= 30:
                        dist = ceil(dist / 2) * 2
                    elif dist <= 80:
                        dist = ceil(dist / 5) * 5
                    else:
                        dist = 80

                    ptime[ag_id] = int(dh.MTM.at[self.name + str(dist) + self.type, AGENT[ag_id]])         
            else:
                ptime[ag_id] = int(dh.MTM.at[self.name, AGENT[ag_id]])
                
        return ptime
    
    def get_tb_time(self, ag_pos:str, ag_id:int) -> int:
        if self.From == 'AGENT':
            if ag_id == 2:
                p1, p2 = sorted([ag_pos, self.To])
                if p1 == p2:
                    return 0
                else:
                    return int(dh.BOTM.at[f"{p1}<->{p2}", "Time"])
            else:    
                dist = dh.cal_dist(dh.POS[self.To], dh.POS[AGENT[ag_id]])
                if dist <= 30:
                        dist = ceil(dist / 2) * 2
                elif dist <= 80:
                    dist = ceil(dist / 5) * 5
                else:
                    dist = 80
                return int(dh.MTM.at[self.name + str(dist) + self.type, AGENT[ag_id]])
            
        return self.time[ag_id]
        
        ## if start position would change
        if self.is_moving_tb():
            # print(AGENT[ag_id], ag_pos, self.From, self.To)
            if ag_id == 2: # BOT
                if self.From == "AGENT" and self.To == "AGENT":
                    print('???')
                elif self.From =="AGENT":
                    p1 = ag_pos
                    p1, p2 = sorted([p1, self.To])
                elif self.To == "AGENT":
                    p2 = ag_pos
                    p1, p2 = sorted([self.From, p2])
                else:
                    p1, p2 = sorted([self.From, self.To])
                    
                if p1 == p2:
                    self.time = 0
                else:
                    self.time = dh.BOTM.at[f"{p1}<->{p2}", "Time"]
                    
                return int(self.time)

            ## LH and RH: Read MTM table
            if self.From == "AGENT":
                dist = dh.cal_dist(dh.POS[self.To], dh.POS[ag_pos])
            elif self.To == "AGENT":
                dist = dh.cal_dist(dh.POS[ag_pos], dh.POS[self.From])
            else:
                dist = dh.cal_dist(dh.POS[self.To], dh.POS[self.From])
            
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
            return int(self.time)
    
    def is_moving_tb(self):
        return self.name in ["R", "M"]


#%% 動素序列
# class RawTherbligs(object):
#     def __init__(self, Pos):
#         self.list = []
#         self.Pos = Pos
#     def __repr__(self):
#         return ", ".join(map(str, self.list))
        
#     def read(self, df: pd.DataFrame):
#         for _, row in df.iterrows():
#             therblig = Therblig(
#                 Name = row["Name"], 
#                 From = row["From"], 
#                 To = row["To"] if not pd.isna(row["To"]) else None, 
#                 Type = row["Type"],
#             )
#             self.list.append(copy.copy(therblig))


#%% 單手任務
class OHT:
    def __init__(self, ls:list):
        self.id:int = -1
        self.tb_list:list = ls
        self.next:list = []
        self.prev:list = []
        self.bind:OHT = None
        self.bind_time:int = 0
        self.end_time:int = 0
        self.To: str
        self.repr_pos:str = self.tb_list[0].To if len(self.tb_list) else ""
        self.type:str = self.decide_type()
        self.time = self.cal_oht_time()
        self.prev_connect_time, self.next_connect_time = self.get_connect_time()
        
        
        # self.is_scheduled = False
            
    def __repr__(self):
        return "(" + ", ".join(map(str, self.tb_list)) + ")"
        # return f"OHT{self.id}"
    
    def decide_type(self) -> str:
        for tb in self.tb_list:
            if tb.name == "A":
                return "A"
            elif tb.name == "DA":
                return "DA"
        return "P&P"
        
    def cal_oht_time(self):
        ptime = [0, 0, 0]
        for ag_id in range(3):
            for tb in self.tb_list:
                ptime[ag_id] += tb.get_tb_time(AGENT[ag_id], ag_id)
        return ptime
    
    def get_connect_time(self):
        prev_connect_time = [0, 0, 0]
        next_connect_time = [0, 0, 0]
        for ag_id in range(3):
            cur_total = 0
            for i, tb in enumerate(self.tb_list):
                cur_total += tb.get_tb_time(AGENT[ag_id], ag_id)
                if tb.name == 'G':
                    prev_connect_time[ag_id] = cur_total
            cur_total = 0
            for i, tb in enumerate(self.tb_list[::-1]):
                if tb.name == 'A' and self.tb_list[i+1].name == 'RL':
                    next_connect_time[ag_id] = cur_total
                    break
                if tb.name == 'RL'and self.tb_list[i-1].name != 'A':
                    next_connect_time[ag_id] = cur_total
                    break
                cur_total += tb.get_tb_time(AGENT[ag_id], ag_id)
        # print(prev_connect_time)
        # print(next_connect_time)
        return prev_connect_time, next_connect_time
                    
    
    def get_oht_time(self, ag_pos, ag_id):
        return self.time[ag_id]
        # print("##### get oht time #####")
        oht_t = 0
        ## Won't go back to origin point when agent is BOT
        if ag_id == 2:
            for tb in self.tb_list:
                oht_t += tb.get_tb_time(ag_pos, ag_id)
        else:
            for tb in self.tb_list:
                oht_t += tb.get_tb_time(ag_pos, ag_id)
        # print("#####")
        return oht_t
    
    ## For AR system
    def get_process_method(self, ag_id) -> dict:
        data = []
        # for tb in self.tb_list:
        #     data.append({
        #         'name': tb.name,
        #         'from': tb.From,
        #         'to': tb.To,
        #         'time': tb.time[2]
        #     })
        # return {self.id: data}
    
        for tb in self.tb_list:
            data.append(dict(
                TaskId = self.id,
                Name = tb.name,
                From = tb.From if tb.From != 'AGENT' else AGENT[ag_id],
                To = tb.To if tb.To != 'AGENT' else AGENT[ag_id],
                time = tb.time[ag_id]
            ))
        return data
    
    def get_bind_remain_time(self, ag_pos, ag_id):
        # print("&&&&& get bind remain time &&&&&")
        rem_t = 0
        for tb in self.tb_list[::-1]:
            if tb.name in ['A', 'DA']:
                break
            rem_t += tb.get_tb_time(ag_pos, ag_id)
        # print("&&&&&")    
        return rem_t
    
    def get_timestamp(self, ag_pos, ag_id):
        # print("!!!!! get bind remain time !!!!!")
        oht_t = 0
        timestamps = []
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(ag_pos, ag_id)
            if tb.is_moving_tb(): 
                if tb.To == 'AGENT':
                    timestamps.append(Timestamp(oht_t, ag_pos))
                else: 
                    timestamps.append(Timestamp(oht_t, tb.To))
        # print("!!!!!")
        return timestamps
    
    def flat(self):
        return self.tb_list
    
    def renew_agent_pos(self, ag_pos_d, ag_id):
        if ag_id != 2:
            return
        # ## find the last movable therblig and change
        for tb in self.tb_list[-2::-1]:
            if tb.name in ['R', 'M']:
                ag_pos_d[ag_id] = tb.To
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
        self.job_list = []
        self.oht_list = []
    
    ## Save tbs by list
    def save_tbs(self):
        for i in range(1, self.num_tbs+1):
            tbs_df = pd.read_excel(f"data/{self.id}_data.xlsx", sheet_name=f"Therbligs{i}")
            tmp = []
            job = []
            for _, row in tbs_df.iterrows():
                if row['Name'] == 'END':
                    oht = OHT(tmp.copy())
                    self.oht_list.append(oht)
                    job.append(oht)
                    tmp.clear() 
                    continue
                
                therblig = Therblig(
                    Name = row["Name"], 
                    From = row["From"] if not pd.isna(row["To"]) else None, 
                    To = row["To"] if not pd.isna(row["To"]) else None, 
                    Type = row["Type"] if not pd.isna(row["To"]) else None, 
                )
                tmp.append(therblig)
            self.job_list.append(JOB(job))
            
        ## Use dummy node to represent "END" OHT   
        self.oht_list.append(OHT([]))                   
           
    def set_oht_id(self):
        for id, oht in enumerate(self.oht_list):
            oht.id = id
            
    def write_process_method(self, ag_id):
        pm = []
        for oht in self.oht_list:
            pm.extend(oht.get_process_method(ag_id))
            
        pm_df = pd.DataFrame(pm)
        pm_df.to_csv(f"./data/{self.id}_process_method_{AGENT[ag_id]}.csv" ,index=False)
     
    def run(self):
        self.save_tbs()
        self.set_oht_id()
        self.write_process_method(2)

#%% Main
# myTBHandler = TBHandler()
# myTBHandler.run()