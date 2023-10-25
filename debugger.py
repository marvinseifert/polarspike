import Overview

overview_df = Overview.Recording.load("D:\Chicken_03_08_21\Phase_01\overview")
cells = []
cells.extend(range(3240))
spikes_df = overview_df.get_spikes_triggered(cells, [0], pandas=False)
print(spikes_df.head)
