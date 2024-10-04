# https://passwd.tistory.com/entry/Python-initpy
# hpe_library/ipywidgets_utils library
from hpe_library.ipywidgets_utils.dropdown import get_dataset_list_dropdown, get_model_list_dropdown
from hpe_library.ipywidgets_utils.button import get_analysis_dh_button, get_analysis_error_button, get_go_to_max_frame_button, get_inference_button, get_procrustes_button, get_reset_button, get_root_rel_button, get_toggle_button, get_visualize_button
from hpe_library.ipywidgets_utils.play import get_play_vis_button
from hpe_library.ipywidgets_utils._select import get_action_select, get_batch_select, get_cam_select, get_part_select, get_part_select2, get_subject_select
from hpe_library.ipywidgets_utils.text import get_batch_num_text, get_error_max_frame_text, get_float_text, get_frame_num_text, get_gts_all_shape_text, get_inputs_all_shape_text, get_results_all_shape_text, get_str_text
from hpe_library.ipywidgets_utils.slider import get_azim_slider, get_delay_slider, get_elev_slider, get_frame_slider, get_trans_slider, get_zoom_slider
from hpe_library.ipywidgets_utils.progress import get_inference_progress

group_dropdown = ['get_dataset_list_dropdown', 'get_model_list_dropdown']
group_button = ['get_analysis_dh_button', 'get_analysis_error_button', 'get_go_to_max_frame_button', 'get_inference_button', 'get_procrustes_button', 'get_reset_button', 'get_root_rel_button', 'get_toggle_button', 'get_visualize_button']
group_play = ['get_play_vis_button']
group__select = ['get_action_select', 'get_batch_select', 'get_cam_select', 'get_part_select', 'get_part_select2', 'get_subject_select']
group_text = ['get_batch_num_text', 'get_error_max_frame_text', 'get_float_text', 'get_frame_num_text', 'get_gts_all_shape_text', 'get_inputs_all_shape_text', 'get_results_all_shape_text', 'get_str_text']
group_slider = ['get_azim_slider', 'get_delay_slider', 'get_elev_slider', 'get_frame_slider', 'get_trans_slider', 'get_zoom_slider']
group_progress = ['get_inference_progress']

__all__ = group_dropdown + group_button + group_play + group__select + group_text + group_slider + group_progress