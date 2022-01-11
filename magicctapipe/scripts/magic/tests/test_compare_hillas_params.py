from compare_hillas_params import compare_hillas_parameters

def test_compare_hillas_params():
	list1 = compare_hillas_parameters(config_file = "compare_hillas_params.yaml", hillas_key="dl1/hillas_params", subrun=5086952, data_or_mc="Data", 
		test_or_train="Test", plot_image=True)
	assert list1 == [False]*len(list1)
	