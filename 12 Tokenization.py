import re, collections

text="a sailor went to sea sea sea "+\
     "to see what he could see see see "+\
     "but all that he could see see see "+\
     "was the bottom of the deep blue sea sea sea"
# text="How much wood could a woodchuck chuck if a woodchuck could chuck wood"
# text="中共中央总书记、国家主席、中央军委主席习近平近日就组建中国资源循环集团有限公司作出重要指示强调，组建中国资源循环集团有限公司，是党中央着眼健全绿色低碳循环发展经济体系，全面推进美丽中国建设作出的重要决策部署。中国资源循环集团有限公司要完整、准确、全面贯彻新发展理念，深入落实全面节约战略，坚持循环利用、变废为宝，坚持创新驱动、开放合作，着力畅通资源循环利用链条，打造全国性、功能性的资源回收再利用平台，推动国民经济循环质量和水平持续提升，为以中国式现代化全面推进强国建设、民族复兴伟业作出积极贡献。"

def InitializeVocabulary(text):
    vocab=collections.defaultdict(int)
    words=text.strip().split()
    for word in words:
        vocab[' '.join(list(word))+' </w>']+=1
    return vocab

vocab=InitializeVocabulary(text)
# print('Vocabulary: {}'.format(vocab))
# print('Size of Vocabulary: {}'.format(len(vocab)))

def GetTokensAndFrequencies(vocab):
    tokens=collections.defaultdict(int)
    for word,freq in vocab.items():
        word_tokens=word.split()
        for token in word_tokens:
            tokens[token]+=freq
    return tokens

tokens=GetTokensAndFrequencies(vocab)
# print('Tokens: {}'.format(tokens))
# print('Number of Tokens: {}'.format(len(tokens)))

def GetPairsAndCounts(vocab):
    pairs=collections.defaultdict(int)
    for word,freq in vocab.items():
        symbols=word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]]+=freq
    return pairs

pairs=GetPairsAndCounts(vocab)
print('Pairs: {}'.format(pairs))
print('Number of distinct pairs: {}'.format(len(pairs)))
most_frequent_pair = max(pairs, key=pairs.get)
print('Most frequent pair: {}'.format(most_frequent_pair))

def MergePairInVocabulary(pair,vocab_in):
    vocab_out={}
    bigram=re.escape(' '.join(pair))
    p=re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
    for word in vocab_in:
        word_out=p.sub(''.join(pair),word)
        vocab_out[word_out]=vocab_in[word]
    return vocab_out

vocab=MergePairInVocabulary(most_frequent_pair, vocab)
print('Vocabulary: {}'.format(vocab))
print('Size of vocabulary: {}'.format(len(vocab)))
tokens=GetTokensAndFrequencies(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))

def Tokenize(text,num_merges):
    vocab=InitializeVocabulary(text)
    for i in range(num_merges):
        tokens=GetTokensAndFrequencies(vocab)
        pairs=GetPairsAndCounts(vocab)
        most_frequent_pair = max(pairs, key=pairs.get)
        print('Most Frequent Pair: {}'.format(most_frequent_pair))
        vocab=MergePairInVocabulary(most_frequent_pair, vocab)
    tokens=GetTokensAndFrequencies(vocab)
    return tokens,vocab

# tokens,vocab=Tokenize(text,num_merges=20)
# print('Tokens: {}'.format(tokens))
# print('Number of tokens: {}'.format(len(tokens)))
# print('Vocabulary: {}'.format(vocab))
# print('Size of vocabulary: {}'.format(len(vocab)))
