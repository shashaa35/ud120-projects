from win10toast import ToastNotifier
from nltk.corpus import stopwords

sw = stopwords.words("english")
print(len(sw))
for word in sw:
	print(word)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print(stemmer.stem("responsiveness"))
# try:
# finally:
# 	toaster = ToastNotifier()
# 	toaster.show_toast("ML Course",
#                    "Task Done",
#                    duration=2)