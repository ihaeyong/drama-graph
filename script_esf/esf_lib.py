#####################################################
# Title: Event Structure Frame list
# Revision_date: 2019. 12. 08 Sunday, 2020. 09. 20
# Author: Seohyun Im
# Contact: seohyunim71@gmail.com
#####################################################

###################################
# The event structure frame list  # 
###################################

# STATE: state, 
# PROCESS: process, cause_process, semelfactive
# MAINTAIN: maintain
# MOTION & CHANGE_OF_LOCATION: move / cause_move, accompany, carry,
#                              move_forward / cause_move_forward, move_back / cause_move_back, move_up / cause_move_up, move_down / cause_move_down, move_toward_speaker,
#                              move_around /cause_move_around, pass / cause_pass,
#                              move_from_source / cause_move_from_source, move_to_goal / cause_move_to_goal, move_from_source_to_goal / cause_move_from_source_to_goal,
# SPREAD: spread / cause_spread, 
# CHANGE_OF_DIRECTION: change_direction / cause_change_direction 
# CHANGE_OF_POSSESSION: lose / cause_lose, get / cause_get, take_cop, give, exchange, 
# INFO_COP: get_info, info_transfer
# CHANGE_OF_STATE: change_state / cause_change_state,
# BECOME: become / cause_become, 
# CHANGE_OF_POSTURE: change_posture / cause_change_posture 
# CHANGE_OF_SCALE: scale_up / cause_scale_up, scale_down / cause_scale_down, scale_move / cause_scale_move
# CHANGE_OF_EXISTENCE: come_into_existence / cause_come_into_existence, go_out_of_existence / cause_go_out_of_existence, 
# EVENT_OCCURRENCE: happen
# ASPECTUAL VERBS: begin / cause_begin, continue / cause_continue, end / cause_end, 
# CAUSATION: negative_causation, positive_causation
# EVENT_ORDER: precede, follow
# SPEECH_ACT: performative

"""
Need to consider the following verb classes:
- change-of-location of liquids: flow, ...
""" 

# pre-state: a presupposed state before the event happen; post-state: an entailed state after the event happen
# d-pre-state: a logically assumed pre-state; d-post-state: a logically assumed post-state
# Semantic Roles (participants of an event): [AGENT, THEME, POSSESSOR, BENEFICIARY, SOURCE_LOCATION, GOAL_LOCATION, SOURCE_STATE, GOAL_STATE, EVENT, STATE, SPEAKER, ADDRESSEE]
# V-ing: the progressive form of the target verb; V-ed: the past-participle form of the target verb
# LOC-PREP: a locative preposition; PREP: a preposition

esfs =[

# STATE
{'etype': 'STATE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'state', 'se_form': 'V-ing(AGENT/THEME, (THEME), (LOCATION))'}]},

# PROCESS
{'etype': 'PROCESS', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, (THEME), (LOCATION))'}]},

{'etype': 'CAUSE_PROCESS', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME)'},
                                   {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(THEME)'}]},

# SEMELFACTIVE    
{'etype': 'SEMELFACTIVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, (THEME))'}]},

# MAINTAIN
{'etype': 'MAINTAIN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed_by(THEME, AGENT)'},
                              {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME)'},
                              {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed_by(THEME, AGENT)'},
                              {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'V-ing(AGENT, THEME)'}]},

# CHANGE-OF-LOCATION    
{'etype': 'MOVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                          {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                          {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},

    
{'etype': 'MOVE_FROM_SOURCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                      {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                      {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(AGENT/THEME, SOURCE_LOCATION)'},
                                      {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_FROM_SOURCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                            {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                            {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(THEME, SOURCE_LOCATION)'},
                                            {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},

    
{'etype': 'MOVE_TO_GOAL', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                  {'se_num': 'se2', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be_not(AGENT/THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                  {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_TO_GOAL', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                        {'se_num': 'se2', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be_not(THEME, GOAL_LOCATION)'},
                                        {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                        {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},

    
{'etype': 'MOVE_FROM_SOURCE_TO_GOAL', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                              {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(AGENT/THEME, GOAL_LOCATION)'},
                                              {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                              {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                                              {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(AGENT/THEME, SOURCE_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_FROM_SOURCE_TO_GOAL', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                                    {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(THEME, GOAL_LOCATION)'},
                                                    {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                                    {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                                    {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(THEME, SOURCE_LOCATION)'}]},


{'etype': 'MOVE_FORWARD', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                  {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                  {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_behind(SOURCE_LOCATION, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_FORWARD', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                        {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                        {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                        {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_behind(SOURCE_LOCATION, GOAL_LOCATION)'}]},

    
{'etype': 'MOVE_BACK', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                               {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                               {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                               {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_behind(GOAL_LOCATION, SOURCE_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_BACK', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                     {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                     {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                     {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_behind(GOAL_LOCATION, SOURCE_LOCATION)'}]},

    
{'etype': 'MOVE_UP', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                             {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                             {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                             {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_higher_than(GOAL_LOCATION, SOURCE_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_UP', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                   {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                   {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                   {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_higher_than(SOURCE_LOCATION, GOAL_LOCATION)'}]},

    
{'etype': 'MOVE_DOWN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                               {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                               {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                               {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_higher_than(SOURCE_LOCATION, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_DOWN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                     {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                     {'se_num': 'se3', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                     {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be_higher_than(SOURCE_LOCATION, GOAL_LOCATION)'}]},
    

{'etype': 'MOVE_TOWARD_SPEAKER', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                         {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(SPEAKER, GOAL_LOCATION)'},
                                         {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                         {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'},
                                         {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(SPEAKER, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_TOWARD_SPEAKER', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                               {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(SPEAKER, GOAL_LOCATION)'},
                                               {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                                               {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'},
                                               {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(SPEAKER, GOAL_LOCATION)'}]},


{'etype': 'MOVE_AROUND', 'esf':  [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                                  {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, FIGURE, SOURCE_LOCATION, GOAL_LOCATION)'},
                                  {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_MOVE_AROUND', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                       {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, FIGURE, SOURCE_LOCATION, GOAL_LOCATION)'},
                                       {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},


{'etype': 'PASS', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_LOCATION)'},
                          {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, PATH, SOURCE_LOCATION, GOAL_LOCATION)'},
                          {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_PASS', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                                {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, PATH, SOURCE_LOCATION, GOAL_LOCATION)'},
                                {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},


{'etype': 'ACCOMPANY', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT, SOURCE_LOCATION)'},
                               {'se_num': 'se2', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                               {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                               {'se_num': 'se4', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT, GOAL_LOCATION)'},
                               {'se_num': 'se5', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},

    
{'etype': 'CARRY', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(AGENT, SOURCE_LOCATION)'},
                           {'se_num': 'se2', 'time': 't1', 'se_type': 'd-pre-state', 'se_form': 'be(THEME, SOURCE_LOCATION)'},
                           {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_LOCATION, GOAL_LOCATION)'},
                           {'se_num': 'se4', 'time': 't2', 'se_type': 'state', 'se_form': 'having(AGENT, THEME)'},
                           {'se_num': 'se5', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(AGENT, GOAL_LOCATION)'},
                           {'se_num': 'se6', 'time': 't3', 'se_type': 'd-post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},


{'etype': 'SPREAD', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME, GOAL_LOCATION)'},
                            {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(AGENT/THEME, GOAL_LOCATION)'},
                            {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, GOAL_LOCATION)'},
                            {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME, GOAL_LOCATION'},
                            {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_LOCATION)'}]},

{'etype': 'CAUSE_SPREAD', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME, GOAL_LOCATION)'},
                                  {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_LOCATION)'}]},

# CHANGE-OF-DIRECTION
{'etype': 'CHANGE_DIRECTION', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME)'},
                                      {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_DIRECTION)'},
                                      {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_DIRECTION, GOAL_DIRECTION)'},
                                      {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME)'},
                                      {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_DIRECTION)'}]},

{'etype': 'CAUSE_CHANGE_DIRECTION', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME)'},
                                            {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_DIRECTION)'},
                                            {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_DIRECTION, GOAL_DIRECTION)'},
                                            {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME)'},
                                            {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_DIRECTION)'}]},

# CHANGE-OF-POSSESSION    
{'etype': 'LOSE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME)'},
                          {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(POSSESSOR, THEME)'},
                          {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'not_have(POSSESSOR, THEME)'}]},

{'etype': 'CAUSE_LOSE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME)'},
                                {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, POSSESSOR, THEME)'},
                                {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'not_have(POSSESSOR, THEME)'}]},

    
{'etype': 'GET', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'not_have(BENEFICIARY, THEME)'},
                         {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(BENEFICIARY, THEME)'},
                         {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'}]},

{'etype': 'CAUSE_GET', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'not_have(BENEFICIARY, THEME)'},
                               {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, BENEFICIARY, THEME)'},
                               {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'}]},

    
{'etype': 'GIVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME)'},
                          {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(POSSESSOR, BENEFICIARY, THEME)'},
                          {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'}]},

    
{'etype': 'TAKE_COP', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME)'},
                              {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(BENEFICIARY, THEME, POSSESSOR)'},
                              {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'}]},

    
{'etype': 'EXCHANGE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME1)'},
                              {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(BENEFICIARY, THEME2)'},
                              {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(POSSESSOR, BENEFICIARY, THEME1, THEME2)'},
                              {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(POSSESSOR, THEME2)'},
                              {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME1)'}]},

    
{'etype': 'GET_INFO', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'not_have(BENEFICIARY, THEME)'},
                              {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(BENEFICIARY, THEME)'},
                              {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'}]},

    
{'etype': 'INFO_TRANSFER', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have(POSSESSOR, THEME)'},
                                   {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(POSSESSOR, BENEFICIARY, THEME)'},
                                   {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(BENEFICIARY, THEME)'},
                                   {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have(POSSESSOR, THEME)'}]},

    
# CHANGE-OF-STATE
{'etype': 'CHANGE_STATE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME, (THEME))'},
                                  {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, (THEME), SOURCE_STATE)'},
                                  {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, (THEME), SOURCE_STATE, GOAL_STATE)'},
                                  {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME, (THEME))'},
                                  {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, (THEME), GOAL_STATE)'}]},

{'etype': 'CAUSE_CHANGE_STATE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME)'},
                                        {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_STATE)'},
                                        {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_STATE, GOAL_STATE)'},
                                        {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME)'},
                                        {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_STATE)'}]},

    
{'etype': 'BECOME', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_STATE)'},
                            {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(AGENT/THEME, GOAL_STATE)'},
                            {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, GOAL_STATE)'},
                            {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_STATE)'},
                            {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(AGENT/THEME, SOURCE_STATE)'}]},

{'etype': 'CAUSE_BECOME', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_STATE)'},
                                  {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not(THEME, GOAL_STATE)'},
                                  {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, GOAL_STATE)'},
                                  {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_STATE)'},
                                  {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not(THEME, SOURCE_STATE)'}]},
    

# CHANGE-OF-POSTURE
{'etype': 'CHANGE_POSTURE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME)'},
                                    {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_POSTURE)'},
                                    {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_POSTURE, , GOAL_POSTURE)'},
                                    {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME)'},
                                    {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_POSTURE)'}]},

{'etype': 'CAUSE_CHANGE_POSTURE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME)'},
                                          {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_POSTURE)'},
                                          {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_POSTURE, , GOAL_POSTURE)'},
                                          {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME)'},
                                          {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_POSTURE)'}]},


# CHANGE-OF-SCALE
{'etype': 'SCALE_UP', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_SCALE)'},
                              {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_SCALE, GOAL_SCALE)'},
                              {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_SCALE)'},
                              {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_higher_than(GOAL_SCALE, SOURCE_SCALE)'}]},

{'etype': 'CAUSE_SCALE_UP', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_SCALE)'},
                                    {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_SCALE, GOAL_SCALE)'},
                                    {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_SCALE)'},
                                    {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_higher_than(GOAL_SCALE, SOURCE_SCALE)'}]},

    
{'etype': 'SCALE_DOWN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_SCALE)'},
                                {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_SCALE, GOAL_SCALE)'},
                                {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_SCALE)'},
                                {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_higher_than(SOURCE_SCALE, GOAL_SCALE)'}]},

{'etype': 'CAUSE_SCALE_DOWN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_SCALE)'},
                                      {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_SCALE, GOAL_SCALE)'},
                                      {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_SCALE)'},
                                      {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_higher_than(SOURCE_SCALE, GOAL_SCALE)'}]},

    
{'etype': 'SCALE_MOVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(AGENT/THEME, SOURCE_SCALE)'},
                                {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME, SOURCE_SCALE, GOAL_SCALE)'},
                                {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(AGENT/THEME, GOAL_SCALE)'}]},

{'etype': 'CAUSE_SCALE_MOVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be(THEME, SOURCE_SCALE)'},
                                      {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME, SOURCE_SCALE, GOAL_SCALE)'},
                                      {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be(THEME, GOAL_SCALE)'}]},

    
# CHANGE-OF-EXISTENCE
{'etype': 'COME_INTO_EXISTENCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME)'},
                                         {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'there_be_not(AGENT/THEME)'},
                                         {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME)'},
                                         {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME)'},
                                         {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'there_be(AGENT/THEME)'}]},

{'etype': 'CAUSE_COME_INTO_EXISTENCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME)'},
                                               {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'there_be_not(THEME)'},
                                               {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME)'},
                                               {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME)'},
                                               {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'there_be(THEME)'}]},

    
{'etype': 'GO_OUT_OF_EXISTENCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'have_not_V-ed(AGENT/THEME)'},
                                         {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'there_be(AGENT/THEME)'},
                                         {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT/THEME)'},
                                         {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'have_V-ed(AGENT/THEME)'},
                                         {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'there_be_not(AGENT/THEME)'}]},

{'etype': 'CAUSE_GO_OUT_OF_EXISTENCE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed(THEME)'},
                                               {'se_num': 'se2', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'there_be(THEME)'},
                                               {'se_num': 'se3', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, THEME)'},
                                               {'se_num': 'se4', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed(THEME)'},
                                               {'se_num': 'se5', 'time': 't3', 'se_type': 'post-state', 'se_form': 'there_be_not(THEME)'}]},

    
# ASPECTUAL VERBS
{'etype': 'BEGIN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_in_progress(EVENT)'},
                           {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(EVENT)'},
                           {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_in_progress(EVENT)'}]},

{'etype': 'CAUSE_BEGIN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_in_progress(EVENT)'},
                                 {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, EVENT)'},
                                 {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_in_progress(EVENT)'}]},

    
{'etype': 'CONTINUE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_in_progress(EVENT)'},
                              {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(EVENT)'},
                              {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_in_progress(EVENT)'}]},

{'etype': 'CAUSE_CONTINUE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_in_progress(EVENT)'},
                                    {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, EVENT)'},
                                    {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_in_progress(EVENT)'}]},

    
{'etype': 'END', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_in_progress(EVENT)'},
                         {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(EVENT)'},
                         {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not_in_progress(EVENT)'}]},

{'etype': 'CAUSE_END', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_in_progress(EVENT)'},
                               {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(AGENT, EVENT)'},
                               {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_not_in_progress(EVENT)'}]},

    
# CAUSATION
{'etype': 'NEGATIVE_CAUSATION', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(AGENT, EVENT)'},
                                        {'se_num': 'se2', 'time': 't2', 'se_type': 'post-state', 'se_form': 'not_happen(EVENT)'}]},

{'etype': 'POSITIVE_CAUSATION', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(AGENT, EVENT)'},
                                        {'se_num': 'se2', 'time': 't2', 'se_type': 'post-state', 'se_form': 'happen(EVENT)'}]},

    
# HAPPEN
{'etype': 'HAPPEN', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'process', 'se_form': 'V-ing(EVENT)'}]},

    
# ORDERING
{'etype': 'PRECEDE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'state', 'se_form': 'V-ing(EVENT1, EVENT2)'},
                             {'se_num': 'se2', 'time': 't1', 'se_type': 'state', 'se_form': 'be_before(EVENT1, EVENT2)'}]},

{'etype': 'FOLLOW', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'state', 'se_form': 'V-ing(EVENT1, EVENT2)'},
                            {'se_num': 'se2', 'time': 't1', 'se_type': 'state', 'se_form': 'be_after(EVENT1, EVENT2)'}]},

    
# SPEECH ACT
{'etype': 'PERFORMATIVE', 'esf': [{'se_num': 'se1', 'time': 't1', 'se_type': 'pre-state', 'se_form': 'be_not_V-ed_by(THEME, ADDRESSEE, SPEAKER)'},
                                  {'se_num': 'se2', 'time': 't2', 'se_type': 'process', 'se_form': 'V-ing(SPEAKER, ADDRESSEE, THEME)'},
                                  {'se_num': 'se3', 'time': 't3', 'se_type': 'post-state', 'se_form': 'be_V-ed_by(THEME, ADDRESSEE, SPEAKER)'}]}
]
