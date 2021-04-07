
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class Ship : UdonSharpBehaviour
{
    
    private Transform SHIP;
    public Transform HUD;

    void Start()
    {
        SHIP = GetComponent<Transform>();
//        HUD = SHIP.transform.FindChild("HUD").gameObject; 

    }

    void Update()
    {
        if (HUD != null) 
        {
            Material HUDMat = HUD.GetComponent<Renderer>().material;
            if (HUDMat != null)
            {
                if (HUDMat.HasProperty("_Dir")) 
                {
                    float pitch,roll,yaw,x,y,z,w;
                    x = SHIP.rotation.x;
                    y = SHIP.rotation.y;
                    z = SHIP.rotation.z;
                    w = SHIP.rotation.w;
                    yaw  = Mathf.Atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z);
                    roll = Mathf.Atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z);
                    pitch   =  Mathf.Asin(2*x*y + 2*z*w);
                    // float pitch = (SHIP.rotation * Quaternion.Euler(180, 0, 0)).eulerAngles.x;
                    // float roll = 180-(SHIP.rotation * Quaternion.Euler(0, 0, 180)).eulerAngles.z;
                    // float yaw = 180-(SHIP.rotation * Quaternion.Euler(180, 0, 0)).eulerAngles.y;
                    //_Dir,_Pos,_Vel,_Target
                    Rigidbody rb = SHIP.GetComponent<Rigidbody>();
                    HUDMat.SetVector("_Dir",new Vector4(pitch,yaw,roll,0.0f));
                    HUDMat.SetVector("_Pos",new Vector4(SHIP.position.x,SHIP.position.y,SHIP.position.z,0.0f));
                    HUDMat.SetVector("_Vel",new Vector4(rb.velocity.x,rb.velocity.y,rb.velocity.z,0.0f));
//                    HUDMat.SetFloat("_Target",new Vector4(0.0f,0.0f,0.0f,0.0f));
                }
            }
        }
    }
}
