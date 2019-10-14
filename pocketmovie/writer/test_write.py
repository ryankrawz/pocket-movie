from pocketmovie.enums import Genre
from writer.double_markov_chain import DoubleMarkov


with open('markov_output.txt', 'w+') as f:
    characters = [
        'Victor',
        'Elizabeth',
        'The Doctor',
        'Jeff',
        'Megan',
    ]
    output_chain = DoubleMarkov(Genre.HORROR, 'The Big Scary', 'Tennessee Williams', characters)
    f.write(output_chain.generate_output())
