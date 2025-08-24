import glob
import os
import json
import opentrons.simulate as simulate
import numpy as np

def pipette_action(protocol, pipette, stock_volume_to_pull, stock_position_to_pull, destination_well, stock_name):
# pipette.transfer(stock_volume_to_pull, stock_position_to_pull, destination_well, new_tip='never') # it might be wise to switch to pipette.aspirate and pipette.dispense, give more control and more modular
    pipette.aspirate(stock_volume_to_pull, stock_position_to_pull)
    if stock_volume_to_pull < 20:
        pipette.air_gap(20 - stock_volume_to_pull)
    elif stock_volume_to_pull < 280:
        pipette.air_gap(20)
    else:
        pass
        #pipette.air_gap(20)
    if stock_name == '1.86mM-PVP-stock':
        protocol.delay(seconds = 7)
        pipette.touch_tip(v_offset = -5, radius = 0.5) 
    #protocol.delay(seconds=3) 
    #pipette.touch_tip(v_offset = -3, radius = 0.5) 
    pipette.dispense(stock_volume_to_pull, destination_well)
    #mix_after=(3, 200)
    #protocol.delay(seconds=3)
    
    
def run_protocol(protocol, directions, loaded_labware_dict):
    protocol.home()
    small_pipette = loaded_labware_dict['Small Pipette']
    small_tiprack = loaded_labware_dict['Small Tiprack']
    large_pipette = loaded_labware_dict['Large Pipette']
    large_tiprack = loaded_labware_dict['Large Tiprack']


    stock_to_pipette_order = directions[0].keys()
    
    if small_pipette.has_tip:
        small_pipette.drop_tip()
    if large_pipette.has_tip:
        large_pipette.drop_tip()

    # module = loaded_labware_dict['Module']
    # module.open_lid()
    # module.set_block_temperature(temperature=50)
     
    for stock_name in stock_to_pipette_order: #THIS LOOP ITERATES OVER THE STOCK SOLUTOINS
        volumes = []
        for stock_index, stock_instructions in directions.items():
            single_stock_instructions = stock_instructions[stock_name]
            stock_volume_to_pull = single_stock_instructions['Stock Volume']
            volumes.append(stock_volume_to_pull)
        
        if small_pipette.max_volume < large_pipette.min_volume:
            raise Exception('Large pipette min volume must be equal or less than small pipette max volume')
            
        small_pipette_dont_pick_up = all(i >= small_pipette.max_volume for i in volumes)
        large_pipette_dont_pick_up = all(i < small_pipette.max_volume for i in volumes)
        all_zero = all(i == 0 for i in volumes)
        
        if small_pipette_dont_pick_up == False and all_zero == False:
            small_pipette.pick_up_tip()
            
        if large_pipette_dont_pick_up == False and all_zero == False:
            large_pipette.pick_up_tip()
             
        for stock_index, stock_instructions in directions.items():
            single_stock_instructions = stock_instructions[stock_name]
            stock_volume_to_pull = single_stock_instructions['Stock Volume']
            stock_position_to_pull = single_stock_instructions['Stock Position']
            destination_well = single_stock_instructions['Destination Well Position']

            if stock_volume_to_pull == 0:
                pass
            elif small_pipette.max_volume <= stock_volume_to_pull:
                pipette = large_pipette
                pipette_action(protocol, pipette, stock_volume_to_pull, stock_position_to_pull, destination_well, stock_name)
            #elif small_pipette.min_volume <= stock_volume_to_pull <= small_pipette.max_volume:
            else:
                pipette = small_pipette
                pipette_action(protocol, pipette, stock_volume_to_pull, stock_position_to_pull, destination_well, stock_name) 

        if small_pipette.has_tip:
            small_pipette.drop_tip()
        if large_pipette.has_tip:
            large_pipette.drop_tip()

    for line in protocol.commands(): 
        print(line)  


def thermocycler_open_lid(protocol, loaded_labware_dict):
    protocol.clear_commands()
    module = loaded_labware_dict['Module']
    module.open_lid()
    for line in protocol.commands(): 
        print(line)  

def thermocycler_close_lid(protocol, loaded_labware_dict):
    protocol.clear_commands()
    module = loaded_labware_dict['Module']
    module.close_lid()
    for line in protocol.commands(): 
        print(line)  

def thermocycler_temperature(protocol, loaded_labware_dict, temp_time, max_volume):
    protocol.clear_commands()
    module = loaded_labware_dict['Module']
    for command in range(temp_time.shape[0]):
        module.set_block_temperature(temp_time[command, 0], hold_time_minutes = temp_time[command, 1], block_max_volume=max_volume)
    
    for line in protocol.commands(): 
        print(line)  

