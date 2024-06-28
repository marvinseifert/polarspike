### 26.06.2024
The cells and stimuli script is not usefull as it is. Better would be to organize all functions that are needed to </br>
organize the input lists from the get_spikes_triggered function in a single module. Ideally the name of this </br>
module should be more clear in signalling its purpose which is to organize the input lists for the get_spikes_triggered </br>
function. The name of the module could be something like get_spikes_organizer.py. </br>

I will come back to this problem when dealing with the get_spikes_triggered function. </br>

### 27.06.2024 </br>
Today, I will try to solve the get_spikes_triggered function. 
I will start by creating a diagram of the function to understand the different steps. </br>
Didnt do the diagram as the function works properly. Will do later. </br>
Now I am working on a carry method and later a derandomization

Carry method: </br>
The problem is that the value in the original cell_df needs to be duplicated according to </br>
how many spikes were found for a given stimulus and then all cells. Maybe I can use the nr_of_spikes column for this. </br>
Other idea: create a multiindex with cell_index and stimulus_index, loc into the dataframe and then adding to the </br>
spike_df. </br>

### 28.06.2024 </br>
Working on carry method continued. 
Have finished the carry method. It works. However, speed performance might suffer. </br>
There might be a way to optimize this function in terms of speed. </br>

I have decided to scrap the derandomization method. It seems to be the wrong place </br>
to implement this here. Rather, there should be a function that will derandomize spikes </br>
using the returned dataframe from the get_spikes_triggered function. </br>

Will finish the bugfixes for now. Remaining tasks:
- stimulus_dfs.py
- stimulus_spikes.py
- stimulus_trace.py
- waveforms.py (which is empty at the moment)