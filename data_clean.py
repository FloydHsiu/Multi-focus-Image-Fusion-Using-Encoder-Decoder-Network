import os

with open('training_fg_names.txt', 'r') as fp:
    txt = fp.read()

fns = txt.split('\n')

os.makedirs('tmp_fg')
os.makedirs('tmp_alpha')

for fn in fns:
    os.rename(os.path.join('.', 'fg', fn), os.path.join('.', 'tmp_fg', fn))
    os.rename(os.path.join('.', 'alpha', fn),
              os.path.join('.', 'tmp_alpha', fn))

os.rename('fg', 'transparent_fg')
os.rename('alpha', 'transparent_alpha')

os.rename('tmp_fg', 'fg')
os.rename('tmp_alpha', 'alpha')
