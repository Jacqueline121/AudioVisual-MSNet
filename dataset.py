from datasets.sum_db import sum_db


def get_training_set(opt, spatial_transform, temporal_transform,
					 target_transform):

	assert opt.dataset in ['movie50', 'tvsum']
	print('Creating training dataset: {}'.format(opt.dataset))

	if opt.dataset == 'tvsum':
		training_data = sum_db(
			opt.video_path_tvsum,
			opt.annotation_path_tvsum_train,
			opt.gt_path_tvsum,
			opt.audio_path_tvsum,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'movie50':
		training_data = sum_db(
			opt.video_path_movie50,
			opt.annotation_path_movie50_train,
			opt.gt_path_movie50,
			opt.audio_path_movie50,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)

	return training_data



def get_test_set(opt, spatial_transform, temporal_transform, target_transform):

	assert opt.dataset in ['movie50', 'tvsum']
	print('Creating testing dataset: {}'.format(opt.dataset))

	if opt.dataset == 'tvsum':
		test_data = sum_db(
			opt.video_path_tvsum,
			opt.annotation_path_tvsum_test,
			opt.gt_path_tvsum,
			opt.audio_path_tvsum,
			spatial_transform,
			temporal_transform,
			target_transform,
			exhaustive_sampling=True)
	elif opt.dataset == 'movie50':
		test_data = sum_db(
			opt.video_path_movie50,
			opt.annotation_path_movie50_test,
			opt.gt_path_movie50,
			opt.audio_path_movie50,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	
	return test_data
