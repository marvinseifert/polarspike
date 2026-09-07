# Script to plot the zebrafish opsins with the multichannel systems LEDs

# Import dependencies
from polarspike import Overview, quality_tests, Opsins, on_off

# Define LEDs (nm)
leds = [660, 610, 560, 535, 500, 460, 416, 365]

# Plot opsins using Opsin template function in Opsins.py
opsin_model = Opsins.Opsin_template()
fig = opsin_model.plot_overview(["Zebrafish"])

# Add vertical lines for each LED wavelength
for led in leds:
    fig.add_vline(
        x=led,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{led}",
        annotation_position="top",
        annotation_font_size=25,
    )

fig.show()
