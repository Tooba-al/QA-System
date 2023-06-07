import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
# from nltk import pos_tag, word_tokenize, RegexpParser 

txt0 = "Sukanya, Rajib and Naba are my good friends. " \
	"Sukanya is getting married next year. " \
	"Marriage is a big step in one’s life." \
	"It is both exciting and frightening. " \
	"But friendship is a sacred bond between people." \
	"It is a special kind of love between us. " \
	"Many of you must have tried searching for a friend "\
	"but never found the right one."

txt1 = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."

txt = "The Panthers finished the regular season with a 15\u20131 record, and quarterback Cam Newton was name the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49\u201315 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 12\u20134 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20\u201318 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl."

stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(txt)

for i in tokenized:
	
	wordsList = nltk.word_tokenize(i)
	wordsList = [w for w in wordsList if not w in stop_words]
	tagged = nltk.pos_tag(wordsList)

	print(tagged)
#///////

# tagged = pos_tag(word_tokenize(txt)) 
# chunker = RegexpParser(""" 
#             NP: {<DT>?<JJ>*<NN>}    #Noun Phrases 
#             P: {<IN>}               #Prepositions 
#             V: {<V.*>}              #Verbs 
#             PP: {<P> <NP>}          #Prepostional Phrases 
#             VP: {<V> <NP|PP>*}      #Verb Phrases 
#                        """)

# output = chunker.parse(tagged) 
# print("After Extracting\n",output)


#////

# blob_object = TextBlob(txt)
# print(blob_object.tags)

#////


