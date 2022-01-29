import pandas as pd
import pylatex as pl

class ToLatex():

    @staticmethod
    def __output_dataframe_to_latex(document, dataframe, file_name = "Dataframe to Latex", caption = "A Table"): 

        with document.create(pl.Section(f"{file_name}")):
            document.append(dataframe.to_latex(header = True, index = True, float_format=".2f", caption = caption))
        document.generate_tex()

    def to_latex(dataframe):
        for config, dataframe_dict in dataframe.items():
                for type_of_data, data in dataframe_dict.items():
                    document = pl.Document(f"{config} - {type_of_data}")
                    ToLatex.__output_dataframe_to_latex(document = document, dataframe= data, caption = f"Table: {config} - {type_of_data}")


