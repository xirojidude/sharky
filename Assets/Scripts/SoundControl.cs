
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class SoundControl : UdonSharpBehaviour
{
    public float beat  = 0.0f;
    public float[] fftL  = new float[256];
    public float[] fftR  = new float[256];
    private float volume =0.0f;

//    public AudioSource m_AudioSource; 

    void Start()
    {
  //      m_AudioSource = this.GetComponent<AudioSource>(); 

    }

    void FixedUpdate()
    {
//        if (this.GetComponent<AudioSource>() != null) {//  (m_AudioSource != null) {
//            m_AudioSource.GetSpectrumData(fftL, 0, FFTWindow.Hamming);
//            m_AudioSource.GetSpectrumData(fftR, 1, FFTWindow.Hamming);
//        }
this.GetComponent<AudioSource>().GetSpectrumData(fftL, 0, FFTWindow.Hamming);
volume = this.GetComponent<AudioSource>().volume;
beat = fftL[0];
if (RenderSettings.skybox.HasProperty("_Beat")) 
{
    RenderSettings.skybox.SetFloatArray("_SoundArray",fftL);
    RenderSettings.skybox.SetFloat("_Beat",beat);
    RenderSettings.skybox.SetFloat("_Volume",volume);
//    Debug.Log("_Beat parameter set to " + beat + " on skybox at volume "+volume);
}
else
{
//    Debug.Log("No _Beats parameter on skybox");
}
// matBlock.SetFloatArray("_ArrayParams",paramData);
// renderer.SetMaterialPropertyBlock(matBlock);
    }
}