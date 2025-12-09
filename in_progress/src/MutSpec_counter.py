import pandas as pd
import os
import seaborn as sns
import Bio
from Bio.Seq import Seq
from Bio import SeqIO
import numpy as np

#Словарь с данными экспериментов на накоплению мутаций
Bacteria_dictionary = {'2740-80_WT_chr1_mod':  ["V. cholerae 2740-80", "2740-80_WT_chr1", 'WT', 'Strand_specific', 1, 2979672, 1545208, 316197],
     '2740-80_WT_chr2_mod':  ["V. cholerae 2740-80", "2740-80_WT_chr2", 'WT', 'Strand_specific', 2, 1101314, 524029, 316197],
     '2740-80_WT_full_mod': ["V. cholerae 2740-80", "2740-80_WT_full", 'WT', 'Strand_specific', 0,  0, 0, 316197],

     '2740-80_mut_chr1_mod':  ["V. cholerae 2740-80", "2740-80_mut_chr1", 'Mut', 'Strand_specific', 1, 2979672, 1545208, 27588],
     '2740-80_mut_chr2_mod':  ["V. cholerae 2740-80", "2740-80_mut_chr2", 'Mut', 'Strand_specific', 2, 1101314, 524029, 27588],
     
     
     'N16961_WT_chr1_mod': ["V. cholerae N16961", "N16961_WT_chr1", 'WT', 'Strand_specific', 1, 2961047, 1564103, 239400],
     'N16961_WT_chr2_mod': ["V. cholerae N16961", "N16961_WT_chr2", 'WT', 'Strand_specific', 2, 1072215, 507982, 239400],
     'N16961_WT_full_mod': ["V. cholerae N16961", "N16961_WT_full", 'WT', 'Strand_specific', 0, 0, 0, 239400],

     'N16961_mut_chr1_mod': ["V. cholerae N16961", "N16961_mut_chr1", 'Mut', 'Strand_specific', 1, 2961047, 1564103, 17100],
     'N16961_mut_chr2_mod': ["V. cholerae N16961", "N16961_mut_chr2", 'Mut', 'Strand_specific', 2, 1072215, 507982, 17100],
      
      
     'ES114_WT_chr1_mod': ["V. fischeri ES114", "ES114_WT_chr1", 'WT', 'Strand_specific', 1, 2897426, 1446504, 248976],
     'ES114_WT_chr2_mod': ["V. fischeri ES114", "ES114_WT_chr2", 'WT', 'Strand_specific', 2, 1330314, 658542, 248976],
     'ES114_WT_full_mod': ["V. fischeri ES114", "ES114_WT_full", 'WT', 'Strand_specific', 0, 0, 0, 248976],

     'ES114_mut_chr1_mod': ["V. fischeri ES114", "ES114_mut_chr1", 'Mut', 'Strand_specific', 1, 2897426, 1446504, 15390],
     'ES114_mut_chr2_mod': ["V. fischeri ES114", "ES114_mut_chr2", 'Mut', 'Strand_specific', 2, 1330314, 658542, 15390],

      
     'R1_WT_chr1_mod': ["D. radiodurans R1", "R1_WT_chr1", 'WT', 'Strand_specific', 1, 1180, 1618525, 256323],
     'R1_WT_chr2_mod': ["D. radiodurans R1", "R1_WT_chr2", 'WT', 'Strand_specific', 2, 412303, 280702, 256323],
     'R1_WT_full_mod': ["D. radiodurans R1", "R1_WT_full", 'WT', 'Strand_specific', 0, 0, 0, 256323],

     'R1_mut_chr1_mod': ["D. radiodurans R1", "R1_mut_chr1", 'Mut', 'Strand_specific', 1, 1180, 1618525, 23832],
     'R1_mut_chr2_mod': ["D. radiodurans R1", "R1_mut_chr2", 'Mut', 'Strand_specific', 2, 412303, 280702, 23832],
      
     'N3610_WT_chr1_mod': ["B. subtilis N3610", "N3610_WT_chr1", 'WT', 'Strand_specific', 1, 1751, 1941719, 375772],
     'N3610_mut_chr1_mod': ["B. subtilis N3610", "N3610_mut_chr1", 'Mut', 'Strand_specific', 1, 1751, 1941719, 158000],

     'MC2_155_WT_chr1_mod': ["M. smegmatis MC2 155", "MC2_155_WT_chr1", 'WT', 'Strand_specific', 1, 1516, 3401143, 4900],

     'HI2424_WT_chr1_mod': ["B. cenocepacia HI2424", "HI2424_WT_chr1", 'WT', 'Strand_specific', 1, 95905, 1808044, 261047],
     'HI2424_WT_chr2_mod': ["B. cenocepacia HI2424", "HI2424_WT_chr2", 'WT', 'Strand_specific', 2, 2813319, 1358611, 261047],
     'HI2424_WT_full_mod': ["B. cenocepacia HI2424", "HI2424_WT_full", 'WT', 'Strand_specific', 0, 0, 0, 261047],

     'PFM2_WT_chr1_mod': ['E. coli PFM2', 'PFM2_WT_chr1', 'WT', 'Strand_specific', 1, 0, 0, 258030],

     'PFM2_mut_chr1_mod': ['E. coli PFM2', 'PFM2_mut_chr1', 'Mut', 'Strand_specific', 1, 0, 0, 258030],

     'IAI1_WT_chr1_mod': ['E. coli IAI1', 'IAI1_WT_chr1', 'WT', 'Strand_specific', 1, 0, 0, 298074],
     'ED1a_WT_chr1_mod': ['E. coli ED1a', 'ED1a_WT_chr1', 'WT', 'Strand_specific', 1, 0, 0, 250674],

      }

#Функция для подсчета числа сайтов (часть скрипта для подсчета частоты мутаций)
def Nucleotide_count(raw): 
    if raw.Base_mutation == 'A:T>G:C' or raw.Base_mutation == 'A:T>T:A' or raw.Base_mutation == 'A:T>C:G':
        raw.Nucleotide_count = raw.A_count+ raw.T_count
    elif raw.Base_mutation == 'G:C>A:T' or raw.Base_mutation== 'G:C>C:G' or raw.Base_mutation== 'G:C>T:A' :
        raw.Nucleotide_count = raw.G_count+ raw.C_count
    elif raw.Base_mutation == 'C>T' or raw.Base_mutation == 'C>G' or raw.Base_mutation == 'C>A':
        raw.Nucleotide_count = raw.C_count
    elif raw.Base_mutation == 'A>G' or raw.Base_mutation == 'A>C' or raw.Base_mutation == 'A>T':
        raw.Nucleotide_count = raw.A_count
    elif raw.Base_mutation == 'T>C' or raw.Base_mutation == 'T>A' or raw.Base_mutation == 'T>G':
        raw.Nucleotide_count = raw.T_count
    elif raw.Base_mutation == 'G>C' or raw.Base_mutation == 'G>A' or raw.Base_mutation == 'G>T':
        raw.Nucleotide_count = raw.G_count
    return raw 


#Скрипт для подсчета частоты мутаций
def MutSpec_counter(List):
    Mut_spec_list= []
    for i in List:
        #создаем колонки Line1 и Mut_spec для 2 мутспеков 
        Mut_spec_12_mod = i[0][['Line','Chromosome', 'Site', 'Base_mutation3']].copy()
        Mut_spec_12_mod['Line1'] = Mut_spec_12_mod.Line+'_mod'
        Mut_spec_12_mod = Mut_spec_12_mod.rename(columns={"Base_mutation3": "Base_mutation"})
        Line_mod = Mut_spec_12_mod.iloc[0].Line+'_mod'

        Mut_spec_6_mod = i[0][['Line','Chromosome', 'Site', 'Base_mutation4']].copy()   
        Mut_spec_6_mod['Line1'] = Mut_spec_6_mod.Line+'_mod'
        Mut_spec_6_mod = Mut_spec_6_mod.rename(columns={"Base_mutation4": "Base_mutation"})
        

        #считаем Mut_counts, создаем общий список для одновременной обработки df 
        Mut_spec_12_mod_counts = Mut_spec_12_mod.groupby(by=['Line1']).Base_mutation.value_counts().to_frame(name='Mutation_count').reset_index()
        Mut_spec_6_mod_counts = Mut_spec_6_mod.groupby(by=['Line1']).Base_mutation.value_counts().to_frame(name='Mutation_count').reset_index()
        Mut_spec_counts_full_list = [Mut_spec_12_mod_counts, Mut_spec_6_mod_counts]

        #Прочтение последовательности и подсчет числа нуклеотидов для шаблона отстающей цепи
        A_count = 0
        G_count = 0
        T_count = 0
        C_count = 0
        for k in i[1]:
            seq_record = SeqIO.read(k[0], "genbank")
            seq_record_seq = seq_record.seq
            OriC_pos = k[1]-1
            Ter_pos = k[2]-1
            if OriC_pos > Ter_pos:
                Strand_specific_fasta = seq_record_seq[:Ter_pos] + seq_record_seq[Ter_pos:OriC_pos].complement() + seq_record_seq[OriC_pos:]
            else:      
                Strand_specific_fasta = seq_record_seq[:OriC_pos].complement() + seq_record_seq[OriC_pos:Ter_pos] + seq_record_seq[Ter_pos:].complement()
            A_count += Strand_specific_fasta.count('A')
            G_count += Strand_specific_fasta.count('G')
            T_count += Strand_specific_fasta.count('T')
            C_count += Strand_specific_fasta.count('C')

        
        #Извлечения данных их словаря Bacteria_dictionary для заполнения таблицы
        Species = Bacteria_dictionary[Line_mod][0]
        Line= Bacteria_dictionary[Line_mod][1]
        Mut_status = Bacteria_dictionary[Line_mod][2]
        Strand = Bacteria_dictionary[Line_mod][3]
        Chromosome = Bacteria_dictionary[Line_mod][4]
        Total_generations = Bacteria_dictionary[Line_mod][7]

        #Создание и заполнение стобцов с информацией
        for i in Mut_spec_counts_full_list:
            i['Species'] = Species
            i['Line']= Line
            i['Mut_status'] = Mut_status
            i['Strand']= Strand
            i['Chromosome'] = Chromosome
            i['A_count']= A_count
            i['G_count']= G_count
            i['T_count']= T_count
            i['C_count']= C_count
            i['Nucleotide_count']=0
            i['Total_generations']= Total_generations
            i['Frequency']=0

        #Подсчет числа мутировавших сайтов
        Mut_spec_12_mod_counts_2 = Mut_spec_12_mod_counts.apply(Nucleotide_count, axis=1)
        Mut_spec_6_mod_counts_2 = Mut_spec_6_mod_counts.apply(Nucleotide_count, axis=1)

        Mut_spec_12_mod_counts_2["Mut_spec"] = '12_mod'
        Mut_spec_6_mod_counts_2['Mut_spec'] = '6_mod'

        Mut_spec_2_list = [ Mut_spec_12_mod_counts_2, Mut_spec_6_mod_counts_2]
        
        #Посчет частоты мутаций мутационного спектра по формуле
        for i in Mut_spec_2_list:
            i['Frequency'] = i['Mutation_count']/i['Nucleotide_count']/i['Total_generations']


        #нормализация частот мутаций 12-компонентного мутационного спектра к 1 
        Mut_spec_12_mod_counts_2_frequency_sum = Mut_spec_12_mod_counts_2.groupby(by= ['Line1']).Frequency.sum().to_frame(name='Frequency_sum').reset_index()
        Mut_spec_12_mod_counts_2 = pd.merge(Mut_spec_12_mod_counts_2, Mut_spec_12_mod_counts_2_frequency_sum, left_on = 'Line1', right_on= 'Line1')
        Mut_spec_12_mod_counts_2['Norm_frequency'] = Mut_spec_12_mod_counts_2.Frequency/Mut_spec_12_mod_counts_2.Frequency_sum

        #нормализация частот мутаций 6-компонентного мутационного спектра к 1 
        Mut_spec_6_mod_counts_2_frequency_sum = Mut_spec_6_mod_counts_2.groupby(by= ['Line1']).Frequency.sum().to_frame(name='Frequency_sum').reset_index()
        Mut_spec_6_mod_counts_2 = pd.merge(Mut_spec_6_mod_counts_2, Mut_spec_6_mod_counts_2_frequency_sum, left_on = 'Line1', right_on= 'Line1')
        Mut_spec_6_mod_counts_2['Norm_frequency'] = Mut_spec_6_mod_counts_2.Frequency/Mut_spec_6_mod_counts_2.Frequency_sum

        ##Бустреп
        bootlist_all = []
        #Bootstrap 12-компонентного спектра
        bootlist_12_mod = []
        for i in range(1000):
            table1 = Mut_spec_12_mod.sample(frac=1, replace = True)
            table2 = table1.groupby(by=['Line']).Base_mutation.value_counts()
            table2 = table2.to_frame(name='Mutation_count')
            table2['iteration']= i
            table2 = table2.reset_index()
            bootlist_12_mod.append(table2)
        bootlist_12_mod_df = pd.concat(bootlist_12_mod, axis=0)
        bootlist_12_mod_df = bootlist_12_mod_df.rename(columns={"Base_mutation3": "Base_mutation"})
        bootlist_12_mod_df['Line1'] = bootlist_12_mod_df.Line +'_mod'
        bootlist_all.append(bootlist_12_mod_df)

        ##Бутстреп 6 компонентного спектра
        bootlist_6_mod=[]
        for i in range(1000):
            table1 = Mut_spec_6_mod.sample(frac=1, replace = True)
            table2 = table1.groupby(by=['Line']).Base_mutation.value_counts()
            table2 = table2.to_frame(name='Mutation_count')
            table2['iteration']= i
            table2 = table2.reset_index()
            bootlist_6_mod.append(table2)
        bootlist_6_mod_df = pd.concat(bootlist_6_mod, axis=0)
        bootlist_6_mod_df = bootlist_6_mod_df.rename(columns={"Base_mutation4": "Base_mutation"})
        bootlist_6_mod_df['Line1'] = bootlist_6_mod_df.Line +'_mod'
        bootlist_all.append(bootlist_6_mod_df)

        #Заполнение данных эксперимента
        for i in bootlist_all:
            i['Species']= Species
            i['Chromosome'] =  Chromosome
            i['Mut_status'] = Mut_status
            i['A_count'] = A_count
            i['G_count'] = G_count
            i['T_count'] = T_count
            i['C_count'] = C_count
            i['Nucleotide_count'] = 0
            i['Total_generations'] = Total_generations
        
        #Подсчет частоты сайтов
        bootlist_12_mod_df = bootlist_12_mod_df.apply(Nucleotide_count, axis=1)
        bootlist_6_mod_df = bootlist_6_mod_df.apply(Nucleotide_count, axis=1)

        #Подсчет частоты для бутстреп-выборки
        bootlist_12_mod_df['Frequency'] = bootlist_12_mod_df.Mutation_count/bootlist_12_mod_df.Nucleotide_count/bootlist_12_mod_df.Total_generations
        bootlist_6_mod_df['Frequency'] = bootlist_6_mod_df.Mutation_count/bootlist_6_mod_df.Nucleotide_count/bootlist_6_mod_df.Total_generations

        
        #Суммы частот для бутстреп-выборки 6-компонентного спектра для нормализации к 1
        bootlist_6_mod_df_frequency_sum  = bootlist_6_mod_df.groupby(by=['Line1', 'iteration']).Frequency.sum().to_frame(name='Frequency_sum').reset_index()
        bootlist_6_mod_df_sum = pd.merge(bootlist_6_mod_df, bootlist_6_mod_df_frequency_sum, left_on=['Line1', 'iteration'], right_on = ['Line1', 'iteration'])

        #Суммы частот для бустреп-выборки 12-компонентого спектра для нормализации к 1
        bootlist_12_mod_df_frequency_sum  = bootlist_12_mod_df.groupby(by=['Line1', 'iteration']).Frequency.sum().to_frame(name='Frequency_sum').reset_index()
        bootlist_12_mod_df_sum = pd.merge(bootlist_12_mod_df, bootlist_12_mod_df_frequency_sum, left_on=['Line1', 'iteration'], right_on = ['Line1', 'iteration'])

        #Нормализация частоты мутаций бутстреп-выборки к 1
        bootlist_12_mod_df_sum['Norm_Frequency'] = bootlist_12_mod_df_sum.Frequency/bootlist_12_mod_df_sum.Frequency_sum
        bootlist_6_mod_df_sum['Norm_Frequency'] = bootlist_6_mod_df_sum.Frequency/bootlist_6_mod_df_sum.Frequency_sum

        #Подсчет стандартного отклонения для бутстреп выборки 6-компонентного мутационного спектра
        bootlist_6_mod_df_sum_std = bootlist_6_mod_df_sum.groupby(by=['Line1', 'Base_mutation']).Frequency.std().to_frame(name='STD').reset_index()
        bootlist_6_mod_df_sum_norm_std = bootlist_6_mod_df_sum.groupby(by=['Line1', 'Base_mutation']).Norm_Frequency.std().to_frame(name='STD_norm').reset_index()
        bootlist_6_mod_df_norm_sum2 = pd.merge(bootlist_6_mod_df_sum_std, bootlist_6_mod_df_sum_norm_std, left_on = ['Line1', 'Base_mutation'], right_on = ['Line1', 'Base_mutation'])


        #Подсчет стандартного отклонения для бутстреп-выборки 12-компонентного мутационного спектра
        bootlist_12_mod_df_sum_std = bootlist_12_mod_df_sum.groupby(by=['Line1', 'Base_mutation']).Frequency.std().to_frame(name='STD').reset_index()
        bootlist_12_mod_df_sum_norm_std = bootlist_12_mod_df_sum.groupby(by=['Line1', 'Base_mutation']).Norm_Frequency.std().to_frame(name='STD_norm').reset_index()
        bootlist_12_mod_df_norm_sum2 = pd.merge(bootlist_12_mod_df_sum_std, bootlist_12_mod_df_sum_norm_std, left_on = ['Line1', 'Base_mutation'], right_on = ['Line1', 'Base_mutation'])

        ##слияние с таблицами для постройки:
        Mut_spec_12_mod_counts_3 = pd.merge(Mut_spec_12_mod_counts_2, bootlist_12_mod_df_norm_sum2, left_on = ['Line1', 'Base_mutation'], right_on = ['Line1', 'Base_mutation'])
        Mut_spec_6_mod_counts_3 = pd.merge(Mut_spec_6_mod_counts_2, bootlist_6_mod_df_norm_sum2, left_on = ['Line1', 'Base_mutation'], right_on = ['Line1', 'Base_mutation'])

        Mut_spec = pd.concat([Mut_spec_12_mod_counts_3, Mut_spec_6_mod_counts_3], axis=0)
        Mut_spec_list.append(Mut_spec)
    Mut_spec_full = pd.concat(Mut_spec_list, axis=0)
    return Mut_spec_full