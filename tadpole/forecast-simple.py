# import sys
# sys.path.append('..')


from os.path import join

from io_local import load_tadpole_data, write_submission_table
from validation import get_test_subjects
from submission import create_submission_table
from models.mixed_model import MixedModel


# Script requires that TADPOLE_D1_D2.csv is in the parent directory.
# Change if necessary.
dataLocationLB1LB2 = '../data/'  # current directory

tadpoleLB1LB2_file = join(dataLocationLB1LB2, 'TADPOLE_LB1_LB2.csv')
output_file = '../data/TADPOLE_Submission_SummerSchool2018_TeamName1.csv'

print('Loading data ...')
LB_table, LB_targets = load_tadpole_data(tadpoleLB1LB2_file)

print('Generating forecasts ...')

# * Create arrays to contain the 84 monthly forecasts for each LB2 subject
n_forecasts = 7 * 12  # forecast 7 years (84 months).
lb2_subjects = get_test_subjects(LB_table)

submission = []
# Each subject in LB2

model = MixedModel()

for rid in lb2_subjects:
    subj_data = LB_table.query('RID == @rid')
    subj_targets = LB_targets.query('RID == @rid')

    # *** Construct example forecasts
    subj_forecast = create_submission_table([rid], n_forecasts)
    subj_forecast = model.create_prediction(subj_data, subj_targets, subj_forecast, rid)

    submission.append(subj_forecast)

## now construct the forecast spreadsheet and output it.
print('Constructing the output spreadsheet {0} ...'.format(output_file))
write_submission_table(submission, output_file)
