import pandas as pd
import panel as pn


class Table:
    def __init__(self, df, width=1000, height=500):
        self.df = df
        self.tabulator = pn.widgets.Tabulator(
            df,
            pagination="remote",
            width=width,
            height=height,
            layout="fit_columns",
        )
        self.column_select = pn.widgets.Select(
            name="Select Column", options=df.columns.to_list()
        )
        self.filter_widget = (
            None  # Placeholder for the widget responsible for filtering
        )
        self.stimulus_select = pn.widgets.Select(
            name="Select Stimulus",
            options=["All"] + df["stimulus_name"].unique().tolist(),
            value="All",
        )
        self.recording_select = pn.widgets.Select(
            name="Select Recording",
            options=["All"] + df["recording"].unique().tolist(),
            value="All",
        )
        self.filter_placeholder = pn.Column()
        # Watch for column selection changes
        self.column_select.param.watch(self.update_filter_widget, "value")

        # Widget for queries
        self.query_widget = pn.widgets.TextInput(
            name="Query", placeholder="Enter query"
        )
        self.query_error = pn.pane.Markdown("", style={"color": "red"})

        # Panel to display the widgets
        self.panel = pn.Column(
            pn.Row(
                pn.Column(
                    self.column_select, self.filter_placeholder
                ),  # Updated layout
                pn.Column(self.stimulus_select),
                pn.Column(self.recording_select),
                pn.Column(self.query_widget, self.query_error),
            ),
            self.tabulator,
            width=width,
            height=height,
        )
        # Initial setup
        self.update_filter_widget()

        self.stimulus_select.param.watch(
            lambda event: self.filter_df(
                self.stimulus_select.value,
                self.recording_select.value,
            ),
            "value",
        )

        self.recording_select.param.watch(
            lambda event: self.filter_df(
                self.stimulus_select.value,
                self.recording_select.value,
            ),
            "value",
        )

        self.query_widget.param.watch(self.apply_filters, "value")

    def show(self):
        return self.panel.servable()

    def update_filter_widget(self, event=None):
        # Remove any previous widget
        self.filter_placeholder[:] = []

        # Get the selected column
        column = self.column_select.value

        # Check the dtype of the column
        if pd.api.types.is_bool_dtype(self.df[column]):
            self.filter_widget = pn.widgets.CheckBoxGroup(
                name=column, options=[True, False], value=[True, False]
            )
            self.filter_widget.param.watch(self.apply_filters, "value")

            # Add the new widget
            self.filter_placeholder.append(self.filter_widget)

        elif pd.api.types.is_numeric_dtype(self.df[column]):
            # Create two FloatInput widgets for lower and upper bounds
            self.lower_bound_input = pn.widgets.FloatInput(
                name=f"Min {column}", value=float(self.df[column].min())
            )
            self.upper_bound_input = pn.widgets.FloatInput(
                name=f"Max {column}", value=float(self.df[column].max())
            )

            # Watch for changes in input values
            self.lower_bound_input.param.watch(self.apply_filters, "value")
            self.upper_bound_input.param.watch(self.apply_filters, "value")

            # Add to filter placeholder
            self.filter_placeholder.extend(
                [self.lower_bound_input, self.upper_bound_input]
            )
        elif pd.api.types.is_string_dtype(self.df[column]):
            print("string")
            options = ["All"] + list(self.df[column].unique())
            self.filter_widget = pn.widgets.MultiSelect(
                name=column, options=options, value=options
            )

            self.filter_widget.param.watch(self.apply_filters, "value")

            # Add the new widget
            self.filter_placeholder.append(self.filter_widget)

    def apply_filters(self, event=None):
        filtered_df = self.df.copy()

        # Boolean
        if pd.api.types.is_bool_dtype(self.df[self.column_select.value]):
            column = self.filter_widget.name
            selected_values = self.filter_widget.value
            filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

        # Numeric filter
        elif pd.api.types.is_numeric_dtype(self.df[self.column_select.value]):
            column = self.column_select.value
            lower = self.lower_bound_input.value
            upper = self.upper_bound_input.value
            filtered_df = filtered_df.loc[
                (filtered_df[column] >= lower) & (filtered_df[column] <= upper)
            ]

        # Categorical filter
        elif pd.api.types.is_string_dtype(self.df[self.column_select.value]):
            column = self.filter_widget.name
            selected_values = self.filter_widget.value
            if "All" not in selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

        # Stimulus filter
        if self.stimulus_select.value != "All":
            filtered_df = filtered_df[
                filtered_df["stimulus_name"] == self.stimulus_select.value
            ]

        # Recording filter
        if self.recording_select.value != "All":
            filtered_df = filtered_df[
                filtered_df["recording"] == self.recording_select.value
            ]

        # Query filter
        try:
            query = self.query_widget.value
            if query != "":
                filtered_df = filtered_df.query(query)
                self.query_error.object = ""
        except KeyError:
            self.query_error.object = "Invalid Key"
        except pd.errors.UndefinedVariableError:
            self.query_error.object = "Invalid Variable, use '' for strings"

        self.tabulator.value = filtered_df

    def filter_df(self, stimulus, recording):
        # Handle cases where lower bound is greater than upper bound

        filtered_df = self.df

        # Further filter based on stimulus_name if not 'All'
        if stimulus != "All":
            filtered_df = filtered_df[filtered_df["stimulus_name"] == stimulus]
        if recording != "All":
            filtered_df = filtered_df[filtered_df["recording"] == recording]

        self.tabulator.value = filtered_df

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
