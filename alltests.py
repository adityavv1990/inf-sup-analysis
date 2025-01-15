import csm

#testlist = ['@x',['0.1', '0.01', '0.001', '0.0001', '0.00001'], '@y', ['standard', 'reduced', 'stabilized'], '@z', ['10', '20', '40', '80', '100', '200', '300', '400','500', '700', '1000']]
testlist = ['@z', ['10', '20', '40', '80', '100', '200', '300', '400','500', '700', '1000']]

mycommands = [ ]

csm.runCombo('clamped.tmpl', 'clamped', testlist, mycommands)
