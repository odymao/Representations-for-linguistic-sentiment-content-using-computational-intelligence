In order to use fasttext, we recommend upload a script on floydhub and generate the embedding matrix for fasttext there.
This is because a bug in the cython version of fasttext (probably not a bug, but we coudldn't import fasttext on our PCs)
the "import fasttext" is now working due to "memroy allocation" problme (in floydhub too)
so we use "import fastText" instead
first you ran main.py of the preprocess project, and after it ends (a word_index.npy got generated inside floyd_aux/) you go on..
So,... inside this folder, execute:
floyd login
floyd init
floyd run --data tsak_auth/datasets/fasttext-en/1:/my_data "git clone https://github.com/facebookresearch/fastText.git && cd fastText && pip install . && cd .. && python floyd_fasttext.py"
