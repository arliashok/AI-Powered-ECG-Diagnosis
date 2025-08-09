import wfdb
import matplotlib.pyplot as plt

# Example: Path to a sample record (adjust the path as needed)
record_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records500/00000/00001_hr'

# Read the record
record = wfdb.rdrecord(record_path)

# Plot the ECG signal
wfdb.plot_wfdb(record=record, title='ECG Record Example')
plt.show()