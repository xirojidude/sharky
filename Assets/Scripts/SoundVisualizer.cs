
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class SoundVisualizer : UdonSharpBehaviour
{
    public SoundControl m_SoundControl;
    private UdonBehaviour m_Udon;
  //  public float[] fftL = new float[256];
  //  public float[] fftR = new float[256];
    public float[] FFTSound = new float[512];
 //   public Renderer rend;
    public Material skybox;

    void Start()
    {
        //rend = GetComponent<Renderer> ();
        skybox = RenderSettings.skybox;
//        m_Udon = m_SoundControl.GetComponent<UdonBehaviour>("UdonBehavior");
       // m_Udon = m_SoundControl.(UdonBehaviour)GetComponent(typeof(UdonBehaviour));
        // Use the Specular shader on the material
        //rend.material.shader = Shader.Find("Specular");

    }

    void Update()
    {
  //      fftL = m_Udon.GetProgramVariable<float[]>("fftL");
  //      fftR = m_Udon.GetProgramVariable<float[]>("fftR");
        m_SoundControl.fftL.CopyTo(FFTSound,0);
        m_SoundControl.fftR.CopyTo(FFTSound,256);

        // Animate the Shininess value
//        float shininess = Mathf.PingPong(Time.time, 1.0f);
        //rend.material.SetFloat("_Color", shininess);
        //rend.material.SetFloatArray("_Sound");
       skybox.SetFloatArray("_Sound",FFTSound);
    }
}
