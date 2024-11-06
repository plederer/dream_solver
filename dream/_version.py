from datetime import datetime
__version__ = "0.1.0"


def version():
    return __version__

def header(title):
        return f"────────────────────────────── {title} ──────────────────────────────"
    
def acknowledgements():

    acknowledgements = [
        f"{header('DREAM SOLVER')}",
        "                                                                          ",
        "Developed by:                                                             ",
        "       Philip Lederer                                                     ",
        "       2022-2023, Institute of Analysis and Scientific Computing, TU Wien,",
        "       2023-2024, Institute of Mathematics, University of Twente,         ",
        "       2024-, Institute of Mathematics, Uni Hamburg.                      ",
        "                                                                          ",
        "       Jan Ellmenreich                                                    ",
        "       2022-, Institute of Analysis and Scientific Computing, TU Wien.    ",
        "                                                                          ",
        "Funding:                                                                  ",
        "       2022-, FWF (Austrian Science Fund) - P35391N                       ",
        "                                                                          ",
        f"Version: {__version__}                                                    ",
        "Github: https://github.com/plederer/dream_solver                          ",
        f"Date:  {datetime.now().strftime('%Hh:%Mm:%Ss %d/%m/%Y')}                  ",
    ]

    return "\n".join(acknowledgements)
