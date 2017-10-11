import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import IO
import model
import plot
import utility


csv_train_sample = str('Category;PalmiticAcid;PalmitoleicAcid;StearicAcid;'
                       'OleicAcid;LinoleicAcid;EicosanoicAcid;'
                       'LinolenicAcid\n'
                       'NA;10.75;0.75;2.26;78.230011;6.72;0.36;0.6\n'
                       'NA;10.88;0.73;2.24;77.084566;7.81;0.31;0.61\n'
                       'NA;9.104569;0.54;2.46;81.129997;5.49;0.31;0.63\n'
                       'NA;10.51;0.67;2.59;77.704569;6.72;0.5;0.8\n'
                       'NA;9.104569;0.49;2.68;79.239998;6.78;0.51;0.7\n'
                       'SA;13.64;2.04;2.25;69.290001;10.84;0.21;0.5\n'
                       'SA;14.1;1.99;2.16;71.300003;9.55;0.21;0.48\n'
                       'SA;13.84;1.78;2.08;71.050003;4.56;0.29;0.67\n'
                       'SA;14.12;1.85;2.17;68.414568;12.03;0.34;0.72\n'
                       'SA;14.1;2.32;2.8;67.150002;12.33;0.32;0.6\n'
                       'U;10.85;0.7;1.8;79.550003;6.05;0.2;0.5\n'
                       'U;10.85;0.7;1.85;79.550003;6;0.25;0.55\n'
                       'U;10.9;0.6;1.9;79.5;6;0.28;0.47\n'
                       'U;10.8;0.65;1.89;79.545698;6.02;0.35;0.2\n'
                       'U;10.9;0.6;1.95;79.550003;6;0.28;0.42\n'
                       'WL;11.9;1.5;2.9;73.400002;10.2;0;0.1\n'
                       'WL;11.1;1.3;2.1;75.5;10;0;0\n'
                       'WL;10.7;1.2;2.1;76;9.845699;0;0.1\n'
                       'WL;10.1;0.9;3.5;74.800003;10.5;0.1;0.1\n'
                       'WL;10.3;1;2.3;77.400002;9;0;0\n')


csv_test_sample = str('Category;PalmiticAcid;PalmitoleicAcid;StearicAcid;'
                      'OleicAcid;LinoleicAcid;EicosanoicAcid;LinolenicAcid'
                      '\n'
                      'NA;9.66;0.57;2.4;79.514567;6.19;0.5;0.78\n'
                      'NA;11;0.61;2.35;77.274569;7.34;0.39;0.64\n'
                      'NA;10.82;0.6;2.39;77.444567;7.09;0.46;0.83\n'
                      'NA;10.36;0.59;2.35;78.68;6.61;0.3;0.62\n'
                      'SA;14.54;1.83;1.96;70.57;10.14;0.27;0.46\n'
                      'SA;13.47;1.94;1.97;72.764567;8.95;0.25;0.46\n'
                      'SA;15.09;2.09;2.57;66.470001;12.4;0.42;0.62\n'
                      'SA;12.86;1.92;2.03;71.32;10.53;0.38;0.65\n'
                      'U;11;0.55;1.98;79.050003;6;0.35;0.5\n'
                      'U;10.85;0.6;1.88;79.550003;6.02;0.3;0.5\n'
                      'U;10.75;0.68;1.95;79.545698;6.02;0.2;0.4\n'
                      'U;10.95;0.6;1.98;79.444567;6;0.38;0.34\n'
                      'WL;10.2;1;2.2;75.300003;10.3;0;0\n'
                      'WL;10.6;1.4;2.4;76.800003;8.3;0.1;0.4\n'
                      'WL;10.6;1.4;2.7;76.145697;8.8;0.1;0.2\n'
                      'WL;11.2;1.3;2.5;75.300003;9.7;0;0\n')


def create_environment():
    try:
        with open('.train_set_synthesis.csv', 'r') as f:
            train_file = f.read()
        if train_file != csv_train_sample:
            raise FileNotFoundError('Train file is different')
    except:
        with open('.train_set_synthesis.csv', 'w') as f:
            f.write(csv_train_sample)

    try:
        with open('.test_set_synthesis.csv', 'r') as f:
            test_file = f.read()
        if test_file != csv_test_sample:
            raise FileNotFoundError('Test file is different')
    except:
        with open('.test_set_synthesis.csv', 'w') as f:
            f.write(csv_test_sample)
