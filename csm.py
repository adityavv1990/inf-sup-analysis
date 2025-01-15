#-----------------------------------------------------------------------------------
#                                        csm.py
#
#    A collection of useful functions for generating and running iris simulations
#    collected by the Computational Solid Mechanics group of IMDEA Materials Institute
#
#                                      March 2021
#
#    Functions:
#
#     checkCases(lists, of)
#     copyFiles(srcDir, desDir, outList): copy files from src to des, but not outList
#     dumpComputerInfo(of)
#     getIrisTimers(): returns assemble, solution and total time in an analysis
#     getIrisWallTime() : returns the wall time spent in a iris analysis
#     getNDofs(filename) : returns the number of unknowns reported in an iris log file
#     getNIterationsAndSteps(fileName) : returns # of iterations & steps in iris solution
#     lastline(filename) : return the lat line of a file
#     readDataFromFile(filename) : return a list of lists with the numerical data
#     runCombo(): runs all possible combinations of options
#     tail(filename, n): return a list with the n last lines of filename
#
#
#   june 2023: updated syntax to python 3.0 (iro)
#-----------------------------------------------------------------------------------

import glob
import os
import platform
import shutil
import subprocess
import sys
import time


#-----------------------------------------------------------------------------------
#                                  checkCases
#-----------------------------------------------------------------------------------
#
# Function to check if the script will run the cases as we actually want
#
# inputs:
#       - lists: list with all test cases explained in previous functions
#       - of: output file
#
# output:
#      - write in the file the list of all cases to be run

def checkCases(lists, of) :
    n = len(lists)
    total_cases = 1
    for i in range(1, n, 2) :
        total_cases *= len(lists[i])
    lineout = 'List of run tests: '

    for i in range(0, total_cases) :
        case = getCombCase(lists, i)
        linei = str(i+1) + '\t' + str(case)
        lineout += '\n' + linei

    of.write(lineout)




#-----------------------------------------------------------------------------------
#                                  copyFiles
#-----------------------------------------------------------------------------------
#
# Function to copy all files in a directory less template and user-pointed files
# inputs:
#       -srcDir : (optional) the source directory where the fiels are stored.
#                   If no srcDir is provided, it will be '.' by defaut, or what
#                   it's the same, current directory.
#       -desDir : destination directory.
#       -outList : LIST with file names that the user does not want to copy
# FUTURE-IMPROVMENTS
#       - Chek if directories are already created on that path, and, if so,
#       rename the new one with the date or another identifier.
def copyFiles(srcDir, desDir, outList) :
    if srcDir == None :
        srcDir='.'

    srcFiles = os.listdir(srcDir) #get the list of all files into the source directory


    for fileOut in outList :
        srcFiles.remove(fileOut)    # kicking out files specified by the user as not to copy


    for fileName in srcFiles :
        fullName = os.path.join(srcDir,fileName)    # joining the path and the file name

        if (os.path.isfile(fullName)) :            # checking if they are regular files.
            shutil.copy( fullName , desDir )            #coping it in the destination




#-----------------------------------------------------------------------------------
#                                dumpComputerInfo
#-----------------------------------------------------------------------------------
#
# Function to get information from the running computer and save it in a file
#
# inputs:
#       - of: output file
#
# output:
#      - cpu, memory information
def dumpComputerInfo():
    myCommand = 'less /proc/cpuinfo > cpuinfo.txt'





#-----------------------------------------------------------------------------------
#                                  getCombCase
#-----------------------------------------------------------------------------------
#
#Function to get a specific case of all posible combinatorics cases from the sets
#given tin the input list (see runComCase fro an exampel of the combinatoric order).3
#
#inputs:
#       -lists: list with the sets of the cases. The lists
#             are defined to ineract with an iris input template as in the following
#             example:  lists=['xx', [set1], 'yy', [set2], ...]
#       - m: is the specific case to be returned in the ordered table with all posible cases.
#             It shifts the sets from the last one to the first one in the entry list.
#output:
#       - casel: a list with the case ready to be write replacing the teamplete and writing
#             the new iris input

def getCombCase(lists , m) :
    n = len(lists)
    casem = [0]*n

    for l in range(0, n, 2):
        casem[l] = lists[l]

    ncase = m
    sizel = 1
    sizel1 = 1
    for l in range(n-1, 0, -2):

        sizel *= len(lists[l])
        k = ncase % sizel
        i = k // sizel1
        casem[l] = lists[l][i]
        sizel1 = sizel

    return casem



#-----------------------------------------------------------------------------------
#                                  getIrisTimers
#-----------------------------------------------------------------------------------
#
# Extracts the time spent if all the phases of an iris.log file
#
# inputs: log file
#
# outputs:
#       an array with the time spent in the assemble, solution, and total
#
# usage: [at, st, tt] = getIrisTimers('iris.log')
#
# note: although there are several ways to read input files in python,
#       we have chosen one way that let us keep the script simple, even
#       when it is not the fastest way to read it. In any case, as
#       files iris.log are not usually too large, it is reasonable to
#       manage as hereunder, laoding all lines in memory.#
#
def getIrisTimers(filename) :
    ll = tail(filename, 5)

    totalTimeline = ll[2]
    timelist = totalTimeline.split()
    counter = 0
    for s in timelist :
        if s == ':' :
            totalTime = timelist[counter + 1]
            break
        counter += 1

    assemblerTimeline = ll[1]
    timelist = assemblerTimeline.split()
    counter = 0
    for s in timelist :
        if s == ':' :
            assemblerTime = timelist[counter + 1]
            break
        counter += 1

    solutionTimeline = ll[0]
    timelist = solutionTimeline.split()
    counter = 0
    for s in timelist :
        if s == ':' :
            solutionTime = timelist[counter + 1]
            break
        counter += 1


    return [assemblerTime, solutionTime, totalTime]




#-----------------------------------------------------------------------------------
#                                  getIrisWallTime
#-----------------------------------------------------------------------------------
#
# Extracts the wall time from an iris.log file
#
# inputs:
#       - there are no inputs
#
# output:
#       -t: double type variable with the time in seconds.
#
def getIrisWallTime(filename) :
    return getIrisTimers(filename)[2]




#-----------------------------------------------------------------------------------
#                                  getNDofs
#-----------------------------------------------------------------------------------
# getNDofs, find the number of unknowns in an iris log file
# usage: n = getNDofs('iris.log')
#
def getNDofs(fileName) :

    with open(fileName) as f:
        lines = f.readlines()
        n = len(lines)
        for l in range( n-1 , 0, -1) :
            if lines[l].find('unknowns') != -1 :
                dofLine = lines[l]
                break

    dofList = dofLine.split()
    ndofs = 0
    for s in range(0, len(dofList), 1) :
        if 'unknowns' in dofList[s] :
            ndofs = dofList[s+2]
            break

    return ndofs




#-----------------------------------------------------------------------------------
#                                  getNIterationsAndSteps
#-----------------------------------------------------------------------------------
# find the number of iterations and steps in an iris log file
# usage: [iter, steps] = getNIterationsAndSteps('iris.log')
#
def getNIterationsAndSteps(fileName) :

    nLines = 13
    ll = tail(fileName, nLines)

    nSteps = 0
    for a in range(0, nLines) :
        stepLine = ll[a]
        if 'steps' in stepLine :
            stepList = stepLine.split()
            counter = 0
            for s in stepList :
                if s == ':' :
                    nSteps = stepList[counter + 1]
                    break
                counter += 1

    nIters = 0
    for a in range(0, nLines) :
        itersLine = ll[a]
        if 'iterations' in itersLine :
            itersList = itersLine.split()
            counter = 0
            for s in itersList :
                if s == ':' :
                    nIters = itersList[counter + 1]
                    break
                counter += 1

    return [nIters, nSteps]




#-----------------------------------------------------------------------------------
#                                  lastLine
#-----------------------------------------------------------------------------------
#
# inputs:
#       - string with the name of the file
#
# output:
#       - a string with the last line of the file
#
def lastLine(filename) :

    with open(filename, 'rb') as f :
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n' :
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()

    return last_line



#-----------------------------------------------------------------------------------
#                                  prepareIrisInputFile
#-----------------------------------------------------------------------------------
#
# Function to replace a list of options and write a irisinput
# inputs:
#       -replaceList : It is a list with pairs of ['word to be substituted, 'new option']
#                      The client who call this function is responsible to list properly
#                      the pairs,
#                      since it is the client the only one who knows the words to replace.
#TO BE DONE. TURN INTO STRING CONSTANTS OR DOUBLE ENTRIES.!!!!!!!!!!!!!!!!
#       -templateName: the name of the file with the iris template.
#
#       -irisName: the name of the '*.iris' that will be run(Ex: muelle.iris).
#                  If it is not defined by the client, the function will use the name
#                  of the template file for the iris name.

def prepareIrisInputfile(replaceList, templateName, irisName) :

    if len(replaceList) % 2 != 0 :                    #check if the list is correct
        print('The list is not set by pairs [keyword,option]')
        return

    if  irisName == None:                           #by defaut, take the name of the template
        irisName=templateName.split('.')[0]         #(just if irisName wasn't defined)

    with open(templateName) as f:                   #Get the whole template
        newText=f.read()


        for i in range(0,len(replaceList),2) :             #Go through the list two by two(by pairs)
            keyword = replaceList[i]
            option = replaceList [i+1]
            while keyword in newText  :
                newText=newText.replace(keyword,option)

    with open(irisName, 'w') as g:
        g.write(newText)




#-----------------------------------------------------------------------------------
#                                  runCombo
#-----------------------------------------------------------------------------------
#
# Function to run several iris analyses and postprocessing, using all possible combinations
# of a list of options
#
# inputs:
#       - templateName: name of the iris file template. This file some special symbols inserted
#               that the python script will replace for numeric values prior to running
#               the actual simulation.
#       - irisName: generic name of the tests. This == just a string that is used to save the
#               the results
#       - options: this == a list of lists are pairs keword-list of options, e.g.
#               runningOptions = ['xx', [5, 10, 15], 'yy', [20, 100], 'zz', ['cg', 'lu']]
#               Running this command with these options will run 3x2x2=12 simulations, taking
#               all possible combinations of the parameters.
#       - extraCommands: list of additional shell commands to run after each iris run, e.g.,
#               extraCommands = ['pvpython defo.py', 'mv defo.png ..', 'rm *.log']
#
# output:
#       - The list of directories where the results are stored
#
# notes:
#       - It will create as many directories as cases, running iris inside each of them
#         followed by the extraCommands
#       - It will also create an irisName.txt file with a summary of output
#       - It will also create an irisName.dat file with only numeric data
#            [5,20,cg]       [10,20,cg]      [15,20,cg]
#            [5,20,lu]       [10,20,lu]      [15,20,lu]
#            [5,100,cg]      [10,100,cg]     [15,100,cg]
#            [5,100,lu]      [10,100,lu]     [15,100,lu]
#
# note:
#       - the iris template file has to be regular file with some 'strange' words,
#         such as xx, yy, zz, whoe value will be replaced in each run case by
#         the data in the runningOptions array
#
def runCombo(templateName, irisName, runningOptions, extraCommands) :

    nOptions = len(runningOptions)
    dirNames = []

    if nOptions % 2 != 0 :
        print('The option list must have an data in pairs [keyword,option]')
        return

    nSimulations = 1
    for i in range(1, nOptions, 2) :
        nSimulations *= len(runningOptions[i])

    # generate input files for simulations
    case = [0] * nOptions
    onlyName = irisName.split('.', 1)[0]
    rootDir = '.'

    infoFile = open(rootDir + '/' + onlyName + '.txt', 'w')
    print("\n------------------------------------------------------------", file = infoFile)
    print("\n           B a t c h   i r i s   s i m u l a t i o n s", file = infoFile)
    print("\n------------------------------------------------------------", file = infoFile)
    print("Local current time :", time.asctime( time.localtime(time.time()) ), file = infoFile)
    print("OS:   ", platform.system(), file = infoFile)
    print("Proc: ", platform.platform(), file = infoFile)
    checkCases(runningOptions, infoFile)
    infoFile.write('\n')

    dataFile = open(rootDir + '/' + onlyName + '.dat', 'w')
    dataFile.write('#    dofs steps iters solu assem total\n')

    for l in range(0, nOptions, 2) :
        case[l] = runningOptions[l]

    # remove old directoress
    dirlist = glob.glob(rootDir + '/' + onlyName + '*.d')
    for d in dirlist:
        try:
            shutil.rmtree(d)
        except:
            print("Error while removing ", d)

    for k in range(0, nSimulations) :
        case = getCombCase(runningOptions, k)

        irisInputFilename = onlyName + str(k+1) + '.iris'
        print("\n   Simulation number: ", k+1, file = infoFile)
        print("   Name: ", irisInputFilename, file = infoFile)
        print("   Options: ", case, file = infoFile)

        dirName = rootDir + '/' + onlyName + str(k+1) + '.d'
        dirNames.append(dirName)
        try:
            os.mkdir(dirName)

        except OSError as e:
            print("Overloading old results for: ", dirName)

        prepareIrisInputfile(case, templateName, irisInputFilename)
        shutil.move(irisInputFilename, dirName)

        str2run = 'iris ' + irisInputFilename
        os.chdir(dirName)
        os.system(str2run)

        n = getNDofs('iris.log')
        print("   Number of dofs : ", n, file = infoFile)

        [niters, nsteps] = getNIterationsAndSteps('iris.log')
        print("   N steps         : ", nsteps, file = infoFile)
        print("   N iterations    : ", niters, file = infoFile)

        timers = getIrisTimers('iris.log')
        print("   Solution time  : ", timers[1], file = infoFile)
        print("   Assemble time  : ", timers[0], file = infoFile)
        print("   Total wall time: ", timers[2], file = infoFile)

        print(k+1, " ", n, " ", nsteps, " ", niters, " ", timers[1], " ", timers[0], " ", timers[2], file = dataFile)

        for cc in extraCommands :
            os.system(cc)

        os.chdir('..')

    infoFile.close()
    return dirNames





#-----3.6 getCasesParameters()
#Function to get the parameters of each case once it was run earlier.
# The output will be store in a lsit wiht the parameter name and its value
#inputs:
#       - keyword: The desired parameter or result to get. It is not the literal word, is will be up to
#                   another specifyed function, in other words, it defines the strategy.
#       - Mainfolder: The parent directory wher are stored as child direcotries with all cases.
#                    if the case was getting from the function runCombinatoricCases() the directory's strucutre
#                   will be as:
#                                       namefolderAllResults
#                                     /       |      \
#                                    /        |       \
#                                  name01   name02    ...
#
#                  note: the name to be pass through is the name before AllResults
#       - templateName: the name of the tempalte, it oculd be got whe OOp is implemented
#output:
#       - wt: list with the time for each case.

def getCasesParameters(keyword,Mainfolder,templateName) :

    #First, compare the template with an iris input, to localize the position of parameters(could be a different function).
    paraLocation = [0,0,0]            # Location of the paramter: [line, column]

    temID = open( templateName , 'r' )
    temStr = temID.readlines()      #Reading the template as an array of lines
    temID.close()

    # Reading the first .iris input
    caseName = 'dirName' + '0'
    os.chdir( caseName )
    irisID = open( caseName + '.iris' , 'r' )
    irisStr = irisID.readlines()   #Reading iris first input
    irisID.close()

    for l in range(0, len( temStr ), 1 ) :           #going through lines
        if temStr[l] != irisStr [l] :

            temList = temStr[l].split(',')                       #separeting by commas
            irisList = irisStr[l].split(',')
            for s in range(0, len(temStr[l]), 1) :
                 print('Work in progress')
           # work in progress.....

    #it is not implemetned so far. but here check or move to the mainfolder directory path
    os.chdir( Mainfolder + 'AllResults')    #Moving to the directory with the results tree

    listdir= os.listdir('.')
    n=len(lisdir)
#    for i in range( 0 , n , 1 ) :
#
#        dirName = mainFolder + str(t)
#        if  dirName in lisdir :
#            os.chdir( dirName )
#
#            with open( dirName + '.iris') as inputID :
#                inputID = inputID.read()
#
#        else :
#            print "No more directories/cases, total: ", i
#                break;




# Function to get some specific parameters from a set document in a tree directory of results adn write them.
# The output will be sotre in a list.
# inputs:
#       - keyword: The desired parameter or result to get. It is not the literal word, is will be up to
#                   another specifyed function, in other words, it defines the strategy.
#       - Mainfolder: The parent directory wher are stored as child direcotries with all cases.
#                    if the case was getting from the function runCombinatoricCases() the directory's strucutre
#                   will be as:
#                                       nameAllResuls
#                                     /       |      \
#                                    /        |       \
#                                  name01   name02    ...
#
# output:
#       - wt: list with the time for each case.


def getData(keyword, mainFolder) :
    #here is defined the strategy, then call to
    #to the specific function that will read the
    #the results. So far, only a strategy for
    #time results == defined
#   if keyword == 'time' :
#        strategy= new timeRobber
#   elseif keyword == "subdivisions"
#   ...
    print(keyword)
    error = 1
    i=0
    os.chdir( mainFolder + 'AllResults')
    listdir= os.listdir('.')
    n=len(listdir)
    maxDirs = len
    wt=[]

    for i in range( 0 , len(listdir) , 1 ) :

        dirName = mainFolder + str(i)   #to get sure that read results in the right order
        if  dirName in listdir :
            os.chdir( dirName )
            #this part is where the strategy will work. With OOP, form the initial creation
            #of the specific strategy class, here just type .get().

            wt.append( getIrisWallTime() )   #wt[i] = Wall Time
            os.chdir( '..' )
        else :
             print("No more directories, total: ", i)
             break;

    os.chdir( '..' )        #coming back to the origanal directory
    return wt




#-----4. runCombinatoricsCluster
#Function to run several cases in iris with the options listed and combined
#   following a combinatory law.
#inputs:
#       -templateName : name of the file template
#       -irisName : name of the main study, as it will iterate an will be many
#                   *.iris files.
#       -lists: this == a list of lists, where are pairs keword-listofoptions
#               as for example:
#                   lits=['xx',[5,10,15],'yy',[20,100,200],'zz',['cg','lu']]
#OUTPUS:
#       -It will create as many directories as cases, the cases are created using
#        combinatorics, using the previous parameter list example:
#            [5,20,cg]       [10,20,cg]      [15,20,cg]
#            [5,20,lu]       [10,20,lu]      [15,20,lu]
#            [5,100,cg]      [10,100,cg]     [15,100,cg]
#            [5,100,lu]      [10,100,lu]     [15,100,lu]
#            [5,200,cg]      [10,200,cg]     [15,200,cg]
#            [5,200,lu]      [10,200,lu]     [15,200,lu]
#       -All outputs from Iris are stored in global directory and then organised case by case
#       in ordered directories.


def runCombinatoricsCluster( templateName , irisName , lists) :

    n=len(lists)                     #get the size of the option list

    if n % 2 != 0   :                 #check if the size of the list is right
        print('The list is not set by pairs [keyword,option]')
        return

    # report the total number of times that will be run Iris
    nSimulations = 1
    for i in range( 1 , n, 2 ) :
        nSimulations *= len(lists[i])

    trline = '#------- Wall Time Recording----------- \n'+ '\n case \t time(s) \n'

    # generating cases.
    case = [0] * n
    onlyName = irisName.split('.',1)[0]

    opf = onlyName + '.txt'       #file where the all cases will be stored
    checkCases( lists ,opf )
    onlyName += 'AllResults'
    try:
        os.mkdir(onlyName)     # creating directory to store current case results
    except OSError as e:
        print('Overloading old results for: ', onlyName)   #If the dir was already created, just overlaod outputs


    shutil.move(opf , onlyName)

    for l in range( 0 , n , 2 ) :
        case [l] = lists [l]


    # generating the iris input.
    for t in range( 0 , nSimulations , 1 ) :
        case=getCombCase(lists,t)


        #writing the input .iris for this case.
        print("--------------------------------\n  ncase", t+1 , "\t", case)
        onlyName=irisName.split('.',1)[0]
        irisNamet = onlyName + str(t)+ '.iris'
        print('runing iris entry named: \t', irisNamet)
        prepareIrisInputfile(case,templateName,irisNamet)  #creating .iris

        dirName= mainName + '/' + onlyName + str(t)

        try:
            os.mkdir(dirName)     # creating directory to store current case results
        except OSError as e:
            print('Overloading old results for: ', dirName)    #If the dir was already created, just overlaod outputs


        shutil.move(irisNamet,dirName)

        str2run= '~/bin/iris ' + irisNamet
        os.chdir(dirName)
        os.system(str2run)                         #running iris
        wallTime = 0         #time variable
        wallTime = getIrisWallTime()       #getIrisWallTime from iris.log
        print('Time for this case was: \t', wallTime)
        trline += '\n' + str(t) + '\t' + str(wallTime)

        os.chdir('../..')       #going back to main dir
        #outList=[templateName]
            #copyFiles('.',dirName,outList)


    with open("time_record.txt", 'w') as g:
        g.write(trline)




#shutil.move("time_record.txt",mainName)

#----------------------- end of function 3.5----------------------------
#-----4.2 runCombinatoricsMeshCluster
#Function to run Iris when the mesh is external and created with gmesh.
#inputs:
#       -templateName : name of the file template
#       -irisName : name of the main study, as it will iterate an will be many
#                   *.iris files.
#       -lists: this is a list of lists, where are pairs keword-listofoptions
#               as for example:
#                   lits=['xx',[5,10,15],'yy',[20,100,200],'zz',['cg','lu']]
#       -meshDir: path to the directory where the meshes are.
#       -meshName: List with the names of the meshes to be used each time.
#               This lsit could hava just one entry(same external mesh for each case)
#               Or it coudl have several entries, in that case the client of this function
#               must match each entry in the list with its related options case.
#OUTPUS:
#       -It will create as many directories as cases, the cases are created using
#        combinatorics, using the previous parameter list example:
#            [5,20,cg]       [10,20,cg]      [15,20,cg]
#            [5,20,lu]       [10,20,lu]      [15,20,lu]
#            [5,100,cg]      [10,100,cg]     [15,100,cg]
#            [5,100,lu]      [10,100,lu]     [15,100,lu]
#            [5,200,cg]      [10,200,cg]     [15,200,cg]
#            [5,200,lu]      [10,200,lu]     [15,200,lu]
#       -All outputs from Iris are stored in global directory and then organised case by case
#       in ordered directories.

def runCombinatoricsMeshCluster( templateName , irisName , lists, meshDir, meshName) :

    n=len(lists)                     #get the size of the option list

    if n % 2 != 0   :                 #check if the size of the list is right
        print('The list is not set by pairs [keyword,option]')
        return

    # report the total number of times that will be run Iris
    nSimulations = 1
    for i in range( 1 , n, 2 ) :
        nSimulations *= len(lists[i])


    trline = '#------- Wall Time Recording----------- \n'+ '\n case \t time(s) \n'

    # generating cases.
    case = [0] * n
    mainName = irisName.split('.',1)[0]

    opf = mainName + '.txt'       #file where the all cases will be stored
    checkCases( lists ,opf )
    mainName += 'AllResults'
    try:
        os.mkdir(mainName)     # creating directory to store current case results
    except OSError as e:
        print('Overloading old results for: ', mainName)   #If the dir was already created, just overlaod outputs


    shutil.move(opf , mainName)

    for l in range( 0 , n , 2 ) :
        case [l] = lists [l]

    # generating the iris input.
    for t in range( 0 , nSimulations , 1 ) :
        case=getCombCase(lists,t)

        #writing the input .iris for this case.
        print('--------------------------------\n  ncase', t+1, '\t', case)
        onlyName=irisName.split('.',1)[0]
        irisNamet = onlyName + str(t)+ '.iris'
        print("runing iris simulation with input file: \t", irisNamet)
        prepareIrisInputfile(case, templateName, irisNamet)

        dirName= mainName + '/' + onlyName + str(t)

        try:
            os.mkdir(dirName)     # creating directory to store current case results
        except OSError as e:
            print('Overloading old results for: ', dirName)    #If the dir was already created, just overlaod outputs


        shutil.move(irisNamet,dirName)

        str2run= '~/bin/iris ' + irisNamet
        os.chdir(dirName)

        if len(meshName) != 1 :
            meshi = meshDir + '/' + meshName[t]
            shutil.copy(meshi,'.')
        else:
            meshi = meshDir + '/' + meshName[0]
            shutil.copy(meshi,'.')

        os.system( str2run )                         #running iris
        wallTime =0         #time variable
        wallTime = getIrisWallTime()
        print("Time for this case was: \t",wallTime)
        trline += '\n' + str(t) + '\t' + str(wallTime)

        os.chdir('../..')       #going back to main dir
        #outList=[templateName]
            #copyFiles('.',dirName,outList)


    with open("time_record.txt", 'w') as g:
        g.write(trline)

#shutil.move("time_record.txt",mainName)


#-----------------------------------------------------------------------------------
#                                  readDataFromFile
#-----------------------------------------------------------------------------------
#
# inputs:
#      - fileName:
#
# output:
#      - a list of lists with floating point numbers

def readDataFromFile(filename):
    with open(filename) as f:
        array = [[float(x) for x in next(f).split()]]
        for line in f: # read rest of lines
            array.append([float(x) for x in line.split()])

    return array



#-----------------------------------------------------------------------------------
#                                  tail
#-----------------------------------------------------------------------------------
# reads the last line(s) of a file
#
# inputs:
#      - fileName:
#      - n: the number of liles to be read
#
# output:
#      - an array, each entry holding a line of the file

def tail(fileName, n=1):
    bs = 1024
    f = open(fileName)
    f.seek(0,2)
    l = 1-f.read(1).count('\n')
    B = f.tell()
    while n >= l and B > 0:
        block = min(bs, B)
        B -= block
        f.seek(B, 0)
        l += f.read(block).count('\n')
    f.seek(B, 0)
    l = min(l,n)
    lines = f.readlines()[-l:]
    f.close()
    return lines
