# CREATE ORDERED PALETTES FOR CLUSTERS

palette = {
    0: '#0000ff',
    1: '#00bfff',
    2: '#00ffbf',
    3: '#40ff00',
    4: '#f2f20d',
    5: '#ff8000',
    6: '#ff0000',
}

palette_7 = {
    0: '#0000ff',
    1: '#00bfff',
    2: '#00ffbf',
    3: '#40ff00',
    4: '#f2f20d',
    5: '#ff8000',
    6: '#ff0000',
}

palette_6 = {
    0: '#0000ff',
    1: '#00bfff',
    2: '#40ff00',
    3: '#f2f20d',
    4: '#ff8000',
    5: '#ff0000',
}

palette_5 = {
    0: '#0000ff',
    1: '#00bfff',
    2: '#f2f20d',
    3: '#ff8000',
    4: '#ff0000',
}

palette_4 = {
    0: '#0000ff',
    1: '#00bfff',
    2: '#ff8000',
    3: '#ff0000',
}

palette_3 = {
    0: '#0000ff',
    1: '#ff8000',
    2: '#ff0000',
}

palette_2 = {
    0: '#0000ff',
    1: '#ff0000',
}

def get_palette(n):
    """There's gotta be a cleaner way"""
    if n == 2:
        return palette_2
    elif n == 3:
        return palette_3
    elif n == 4:
        return palette_4
    elif n == 5:
        return palette_5
    elif n == 6:
        return palette_6
    elif n == 7:
        return palette_7