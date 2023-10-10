import pandas as pd
import panel as pn


class Table:
    def __init__(self, df):
        self.df = df
        # Tabulator widget with remote pagination
        self.tabulator = pn.widgets.Tabulator(df, pagination='remote')

        # FloatInput widgets for lower and upper bounds
        self.lower_bound_input = pn.widgets.FloatInput(name='Lower Bound')
        self.upper_bound_input = pn.widgets.FloatInput(name='Upper Bound')

        # Select widget for column selection
        options = list(filter(lambda x: x not in ["stimulus_name", "recording"], df.columns.to_list()))

        self.column_select = pn.widgets.Select(
            name='Select Column',
            options=options,
            value=options[0]
        )

        # Select widget for stimulus_name
        self.stimulus_select = pn.widgets.Select(
            name='Select Stimulus',
            options=['All'] + df['stimulus_name'].unique().tolist(),
            value='All'
        )

        self.recording_select = pn.widgets.Select(
            name='Select Recording',
            options=['All'] + df['recording'].unique().tolist(),
            value='All'
        )

        self.initialize_inputs()
        self.column_select.param.watch(self.update_input_range, 'value')

        # Callback function to filter the DataFrame

        self.lower_bound_input.param.watch(
            lambda event: self.filter_df(self.lower_bound_input.value, self.upper_bound_input.value, self.column_select.value,
                                    self.stimulus_select.value, self.recording_select.value), 'value')
        self.upper_bound_input.param.watch(
            lambda event: self.filter_df(self.lower_bound_input.value, self.upper_bound_input.value, self.column_select.value,
                                    self.stimulus_select.value, self.recording_select.value), 'value')
        self.stimulus_select.param.watch(
            lambda event: self.filter_df(self.lower_bound_input.value, self.upper_bound_input.value, self.column_select.value,
                                    self.stimulus_select.value, self.recording_select.value), 'value')

        self.react = pn.bind(self.filter_df, self.lower_bound_input, self.upper_bound_input, self.column_select,
                             self.stimulus_select)

        self.recording_select.param.watch(
            lambda event: self.filter_df(self.lower_bound_input.value, self.upper_bound_input.value, self.column_select.value,
                                    self.stimulus_select.value, self.recording_select.value), 'value')

        self.react = pn.bind(self.filter_df, self.lower_bound_input, self.upper_bound_input, self.column_select,
                             self.stimulus_select, self.recording_select)  # Add the recording select value here

        # Panel to display the widgets
        self.panel = pn.Column(
            pn.Row(pn.Column(self.column_select,
                             self.lower_bound_input,
                             self.upper_bound_input),
                   self.stimulus_select,
                   self.recording_select),  # Add the recording select widget here
            self.tabulator,
            self.react
        )

    def show(self):
        # Serve the panel
        return self.panel.servable()

    def get_filtered_df(self):
        return self.tabulator.value

    def get_selected_rows(self, event):
        selected_indices = self.tabulator.selection
        if selected_indices:
            selected_rows_df = self.tabulator.value.iloc[selected_indices]
            return selected_rows_df
        else:
            return None

    def get_changed_df(self):
        return self.tabulator.value

    def initialize_inputs(self):
        column = self.column_select.value
        self.lower_bound_input.value = float(self.df[column].min())
        self.upper_bound_input.value = float(self.df[column].max())

    # Callback function to update the FloatInput range when a column is selected
    def update_input_range(self, event):
        self.initialize_inputs()
        self.filter_df(self.lower_bound_input.value, self.upper_bound_input.value, self.column_select.value,
                  self.stimulus_select.value)  # Update the tabulator after changing the input range

    def filter_df(self, lower_bound, upper_bound, column, stimulus, recording):
        # Handle cases where lower bound is greater than upper bound
        lower = min(lower_bound, upper_bound)
        upper = max(lower_bound, upper_bound)

        filtered_df = self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]

        # Further filter based on stimulus_name if not 'All'
        if stimulus != 'All':
            filtered_df = filtered_df[filtered_df['stimulus_name'] == stimulus]
        if recording != 'All':
            filtered_df = filtered_df[filtered_df['recording'] == recording]

        self.tabulator.value = filtered_df






