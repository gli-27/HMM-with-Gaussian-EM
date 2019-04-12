#!/usr/bin/env python3
import sys
import os

DATA_DIR = '../data/' #TODO: change this to wherever you put the data if working on a different machine
SIGFIG_NUM = 5

def err(msg):
    print('ERROR: {}'.format(msg), file=sys.stderr) #NOTE: If you get a SyntaxError on this line, you are using Python 2, which is wrong. Use Python 3.
    exit()

def find_filenames():
    return [fn for fn in os.listdir('.') if os.path.isfile(fn) and (fn.endswith('_hmm_gaussian.py') or fn.endswith('_hmm_aspect.py'))]

def get_output(filename):
    import subprocess
    cmd = None
    if filename.endswith('_hmm_gaussian.py'):
        cmd = './{} --nodev --iterations 2 --clusters_file gaussian_hmm_smoketest_clusters.txt --data_file {} --print_params'.format(filename, os.path.join(DATA_DIR,'points.dat'))
    else:
        cmd = './{} --nodev --iterations 2 --clusters_file aspect_hmm_smoketest_clusters.txt --data_file {} --print_params'.format(filename, os.path.join(DATA_DIR,'pairs.dat'))
    print('Running this command:\n{}'.format(cmd))
    try:
        output = subprocess.check_output(cmd.split()).decode('utf-8')
    except subprocess.CalledProcessError:
        err('Python file did not exit successfully (likely crashed).')
    except OSError as e:
        if e.errno == 13:
            err('Python file is not executable; run this command:\nchmod u+x {}'.format(filename))
        elif e.errno == 8:
            err('Python file does not start with a shebang; put this line at the very top:\n#!/usr/bin/python3')
        elif e.errno == 2:
            err('Unable to execute python file; if you ever edited the file on Windows, it is possible that the line endings are wrong, so try running this command:\ndos2unix {}\nOtherwise, you\'ll have to take a look at it and see what went wrong.'.format(filename))
        else:
            print('Got an OS error not caused by permissions or a shebang problem; you\'ll have to take a look at it and see what the problem is. See below:')
            raise e

    return output

def tokens(s):
    result = []
    for tok in s.split():
        try:
            result.append(float(tok))
        except ValueError as e:
            result.append(tok)
    return result

def round_to_sigfigs(num, sigfigs):
    from math import log10, floor
    if num == 0:
        return num
    else:
        return round(num, -int(floor(log10(abs(num)))) + sigfigs - 1)

def fuzzy_match(line, req):
    line_toks = tokens(line)
    if len(line_toks) != len(req):
        return False
    else:
        for l,r in zip(line_toks, req):
            if type(l) != type(r):
                return False
            elif type(l) == str and l != r:
                return False
            elif type(l) == float and round_to_sigfigs(l,SIGFIG_NUM) != round_to_sigfigs(r,SIGFIG_NUM): #float
                return False
        return True

class Req:
    def __init__(self, req, name):
        self.req = tokens(req)
        self.name = name
        self.matched = False

    def check(self,line):
        if fuzzy_match(line, self.req):
            self.matched = True

    def report(self):
        s = '{}: '.format(self.name)
        if self.matched:
            return s + 'Correct!'
        else:
            return s + 'NOT CORRECT!'

    def req_str(self):
        return ' '.join(map(str,self.req))


def verify_reqs(reqs, output):
    for line in output.split('\n'):
        for r in reqs:
            r.check(line)
    for r in reqs:
        print(r.report())
    if not all([r.matched for r in reqs]):
        err('Unable to find one or more required output lines. Make sure each is on its own line and formatted correctly; if so, then there is an implementation problem. This should have produced (with all numbers matched to {} significant figures):\n{}\n'.format(SIGFIG_NUM, '\n'.join([r.req_str() for r in reqs])))

def main():
    filenames = find_filenames()
    if len(filenames) == 0:
        err('No files ending in \'_hmm_gaussian.py\' or \'_hmm_aspect.py\' found. Make sure your file is named LastName_hmm_gaussian.py or LastName_hmm_aspect.py.')
    if len(filenames) > 1:
        err('Only include a single file ending in \'_hmm_gaussian.py\' or \'_hmm_aspect.py\' in the submission directory.')
    print('Found Python file to run.')
    if not os.path.exists(DATA_DIR):
        err('Could not find the data directory; looked for {}. Change the DATA_DIR variable at the top of this smoke test file to wherever you have downloaded the data (points.dat or pairs.dat).'.format(DATA_DIR))
    print('Found data directory.')
    output = get_output(filenames[0])
    print('Ran Python file.')

    reqs = None
    if filenames[0].endswith('_hmm_gaussian.py'):
        reqs = [
            Req('Gaussian','Choice of Gaussian vs. Aspect'),
            Req('Train LL: -4.7431560430438005', 'Training average log-likelihood'),
            Req('Initials: 4.536917830049958e-06 | 0.9999954630821699','Initials'),
            Req('Transitions: 0.6504342951696082 0.34787620154165316 | 0.28894829172105657 0.7104186916567568', 'Transitions'),
            Req('Mus: -0.7516214514802897 -0.48028441694576895 | -0.8929993723943579 -0.6872117204126011', 'Mus'),
            Req('Sigmas: 2.061572429505781 0.8046007515753574 0.8046007515753574 7.526702888938909 | 12.475372700630857 -0.9590967624670685 -0.9590967624670685 5.300042768142592', 'Sigmas')
            ]
    else:
        reqs = [
            Req('Aspect','Choice of Gaussian vs. Aspect'),
            Req('Train LL: -4.494342637270131', 'Training average log-likelihood'),
            Req('Initials: 0.023146429113232822 | 0.9768535708867673','Initials'),
            Req('Transitions: 0.5947249610519026 0.4038144905206853 | 0.4068622046905963 0.5923775344442254','Transitions'),
            Req('Theta_1: 0.08985890375612285 0.1947256104123401 0.03787114173455456 0.17759038483399966 0.10249112012175535 0.06835554679181831 0.08651171756839614 0.07171633810381216 0.11477095954712623 0.056108277130075775 | 0.16584997457920977 0.12736069352099177 0.0777649051555383 0.09112404593377416 0.15093996563947915 0.02711680374174635 0.10018253190811484 0.05268971743072799 0.17201119743034005 0.034960164660076745','Theta_1'),
            Req('Theta_2: 0.15042728384812828 0.18165994252355863 0.0587281273780455 0.06003236677806756 0.06025185870482323 0.10007035455685982 0.11098674714970497 0.08677789992129131 0.0922769548002609 0.09878846433926074 | 0.07831611050528237 0.09817160790197837 0.10581131408365732 0.10227508613094947 0.11318828138078005 0.16005066423801428 0.0911955435078659 0.08210155377498105 0.08771383846434914 0.08117600001214123','Theta_2')
                ]
    verify_reqs(reqs,output)
    print('Congratulations, you passed this simple test! However, make sure that your code runs AND PASSES THIS TEST on the csug server. Also, don\'t forget your README!')

if __name__ == '__main__':
    main()
