stats_help = '''\
normalization:
        F: '(delta F - F0/F0)'
        Z: 'calcium dependant Z-score - isosbestic Z-score'
standardization:
        Zsc: 'Z-score with preevent baseline mean and standard deviation'''


behavior_help = '''\
#################################################################################################################################################################
                                                                    FULL HELP
#################################################################################################################################################################
There are two main types of data, EVENTS and INTERVALS. EVENTS are always lowercase and INTERVALS are uppercase.
For all ON/on commands, corresponding variables with OFF/off exist (eg: 'hled_off' or <obj>.LED1_OFF).
<obj> is the name of the object, ie the user chosen name when creating the object (eg: data = <this_file>.BehavioralData(XX), in that case <obj> is data). 

NB: 
- For all switch (on/off) commands, unnecessary commands are also listed (eg.: if 2 switch off commands occurs in a row, the second one is in the list)
- For all intervals, unnecessary commands are accounted for in order to obtain the real ON/OFF intervals.

EVENTS: [<obj>.events()]
****************************************************************************************************************************************************************** 
 - <obj>.hled_on     : houselight switch on command  
 - <obj>.led1_on     : LED1 switch on command  
 - <obj>.led2_on     : LED2 switch on command   
 - <obj>.np1         : nosepoke hole 1 visited  
 - <obj>.np2         : nosepoke hole 2 visited  
 - <obj>.inj1        : injection occurs (when injection consist of multiple pump turns, only the first one is registered)  
 - <obj>.ttl1_on     : TTL1 starts
 - <obj>.rec_start   : timestamp at which recording TTL1 switches on first  
 - <obj>.np1_<n>     : active nosepoke number n (modulo FR) ; eg. in the case of an experiment with an FR = 5, <obj>.np1_5 is the rewarded nosepoke 
 
SWITCHES:
******************************************************************************************************************************************************************
- <obj>.switch_d_nd  : switch from 'drug period' to 'no drug period' if LED2 is ON before the switch
- <obj>.switch_to_nd : switch from 'drug period' to 'no drug period' if timeout before the switch
- <obj>.switch_nd_d  : switch from 'no drug period' to 'drug period'
 
INTERVALS: [<obj>.intervals()]
****************************************************************************************************************************************************************** 
 - <obj>.HLED_ON     : period with houselight on
 - <obj>.TTL1_ON     : period with TTL1 on (ie: recorded period in settings where TTL1 controls recording)
 - <obj>.LED1_ON     : period with LED1 ON
 - <obj>.LED2_ON     : period with LED2 OFF
 - <obj>.DARK        : period with all lights off (LED1, LED2 and HLED)
 - <obj>.TO_DARK     : period with all lights off occuring near an injection thus being part of a timeout
 - <obj>.TIMEOUT     : combination of LED1_ON and TO_DARK, ie full timeouts
 - <obj>.NOTO_DARK   : period of dark not included in a timeout (can be first 10s of darkness in some settings)
 - <obj>.D_<n>       : 'drug period' number n, in settings where a period off approximately 40 min with HLED off is defined as a drug period   
 - <obj>.ND_<n>      : 'no drug period' number n, in settings where a period off approximately 15 min with HLED off is defined as a no drug period      

DICTIONNARY:
****************************************************************************************************************************************************************** 
Some elements can also be specified by using <obj>.dictionnary("<name_of_element>") or directly by "<name_of_element>" in the case of user functions. 
Full list of variables:
- events  : 'np1', 'np2', 'inj1'
- periods : 'HLED_ON', 'HLED_OFF', 'TTL1', 'LED1_ON', 'LED1_OFF', 'LED2_OFF','LED2_ON', 'DARK', 'TO_DARK', 'NOTO_DARK', 'TIMEOUT'
This list can be obtained by running <obj>.elements

USER FUNCTIONS:
****************************************************************************************************************************************************************** 
<obj>.get(names,id_tuple)
    This function returns the raw data, filtered by either the name of the imetronic operator (full list in the config.yaml file or the IMETRONIC manual) or its tuple (eg: (6,1) for injection 1)

<obj>.graph(element,optional_param)
    This function shows an eventplot of the element (which can be inputed as a string or as data (see above)).

<obj>.summary()
    This function automatically graphs useful events and intervals, thus giving a preview of the dat file content. The list can be modified in config.yaml

<obj>.timestamps(events,interval,intersection,exclude,<additional_parameters>)
    This function returns the timestamps for specified events occuring during specified intervals. All event and interval parameteres must be inputed as lists (in brackets ; eg. interval=[<obj>.ND[1]]) except for single elements in string form (eg. events='np1').
    events        : only non optional parameter                  ; example : events = ['np1'] (ie. all active nosepokes)
    interval      : optional, (default is whole recording)       ; example : interval = ['HLED_ON'] (ie. intervals with LED1 on)
    intersection  : optional, intersection of multiple intervals ; example : intersection = ['TIMEOUT', <obj>.D[1]] (ie. first drug period timeouts)
    exclude       : optional, interval to disregard              ; example : exclude = [<obj>.ND[2]] (ie. exclude no drug period 2)
    additional parameters (default setting in parenthesis):
     - to_csv (True)        : output a csv file
     - graph  (True)        : outputs agraphical representation
     - filename ('default') : filename of the outputed csv (default is dat file name)
     - start_TTL1 (True)    : correct timestamps to align them with fiber data

ADDITIONAL ATTRIBUTES:
****************************************************************************************************************************************************************** 
 - <obj>.df          : full dat file as panda DataFrame 
 - <obj>.filepath    : filepath of dat file represented by the object 
 - <obj>.end         : last timestamp of the recording
 
 OPTIONAL ATTRIBUTES:
****************************************************************************************************************************************************************** 
 The following attributes can be modified when creating the object, for example: <objname> = <this_file>.BehavioralData(<filepath>,TO_duration=<choosen_value>)
 - <obj>.fixed_ratio : fixed ratio 
 - <obj>.rat_ID      : optional rat_ID (eg. rat number) 
'''