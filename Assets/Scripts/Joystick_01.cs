
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class Joystick_01 : UdonSharpBehaviour
{

    //[SerializeField] Transform Ship;
    public Transform Ship;
    public Transform SpawnLocation;
    public Transform JoySpawnLocation;
    public Animator animator;

    public float speed = 0.0f;
    public float vspeed = 0.0f;
    public int mode = 0;
    public float maxSpeed = 50.0f;
    public float acceleration = 0.005f;
    public float deceleration = 0.005f;
    public bool isOff = true;
    private float thrust = 0.0f;
    private VRCPlayerApi player;
    private float smooth = 2.1f;
    private Vector3 joystickSpawnPosition;
    private Quaternion joystickSpawnRotation;
    private Rigidbody m_Rigidbody;
    private float dropTime=0.0f;
    public int dropreset=0;
 
    Quaternion JoystickZeroPoint;
    private float roll = 0f;
    private float pitch = 0f;
    private float yaw = 0f;

    [System.NonSerializedAttribute] [UdonSynced(UdonSyncMode.Linear)] public Vector3 RotationInputs;
    [System.NonSerializedAttribute] public bool Piloting = false;
    [System.NonSerializedAttribute] public bool InEditor = true;
 //   [System.NonSerializedAttribute] public bool InVR = false;
    [System.NonSerializedAttribute] public bool Passenger = false;
    [System.NonSerializedAttribute] public Vector3 LastFrameVel = Vector3.zero;

    void Start()
    {
        player = Networking.LocalPlayer;
        Networking.SetOwner(player, gameObject);
//        if (player.IsUserInVR()) { InVR = true; }
//        joystickSpawnPosition = new Vector3(1.7899f,-0.5435001f,0.03769994f);
        joystickSpawnPosition = this.transform.localPosition;
        joystickSpawnRotation = this.transform.localRotation;
        m_Rigidbody = Ship.GetComponent<Rigidbody>();
        animator.SetTrigger("isOff");
        animator.ResetTrigger("isAccelerating");
        animator.ResetTrigger("isDecelerating");
        animator.ResetTrigger("isHighSpeed");
        animator.ResetTrigger("isHover");
        isOff=true;
    }
//    void Update() {
    void FixedUpdate() {

        if (speed >0.0f) 
        {
        Ship.position += transform.forward * Time.deltaTime * speed;
        Quaternion Q = Ship.rotation;
        //  From a quaternion ...
          float shipyaw  = Mathf.Atan2(2*Q.y*Q.w - 2*Q.x*Q.z, 1 - 2*Q.y*Q.y - 2*Q.z*Q.z);
          float shiproll = Mathf.Atan2(2*Q.x*Q.w - 2*Q.y*Q.z, 1 - 2*Q.x*Q.x - 2*Q.z*Q.z);
          float shippitch   =  Mathf.Asin(2*Q.x*Q.y + 2*Q.z*Q.w);

        //float pitch = (transform.rotation * Quaternion.Euler(180, 0, 0)).eulerAngles.x;
        //float roll = 180-(transform.rotation * Quaternion.Euler(0, 0, 180)).eulerAngles.z;
 //       Debug.Log("ShipPitch:" + shippitch + "  ShipRoll:" + shiproll + "  ShipHeading:" + shipyaw + ")");

        // Quaternion bodyDirection =player.GetRotation();
        // Vector3 bodyVector = bodyDirection.eulerAngles;
        // Debug.Log("body(" + bodyVector.x + ", " + bodyVector.y + ", " + bodyVector.z + ")");

    //    Vector3 shipVector = Ship.rotation.eulerAngles;
//        Debug.Log("ship(" + Ship.rotation.eulerAngles.x + ", " + Ship.rotation.eulerAngles.y + ", " + Ship.rotation.eulerAngles.z + ") LocalTransforn(" + transform.localEulerAngles.x + ", " + transform.localEulerAngles.y + ", " + transform.localEulerAngles.z + ")");

        // Quaternion headDirection = player.GetBoneRotation( HumanBodyBones.Head );
        // Vector3 headVector = headDirection.eulerAngles;
        // Debug.Log("head(" + headVector.x + ", " + headVector.y + ", " + headVector.z + ")  Transforn(" + transform.rotation.x + ", " + transform.rotation.y + ", " + transform.rotation.z + ")");


//        Quaternion newDirection = Quaternion.FromToRotation(shipVector,headVector);
        Quaternion newDirection = transform.localRotation; 
        // Roll, Yaw, Pitch
        newDirection *= Quaternion.Euler(-shiproll*180.0f/3.14159f, -90f, 15f); // make 0,0,0 be center of players view. compensate for 2 degrees of roll drift

    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
//    rd.xy = mul(rot2( path(lookAt.z).x/16. ),rd.xy);


        //newDirection = Quaternion.Slerp(newDirection, Quaternion.Euler(transform.rotation.x, newDirection.y, newDirection.z) , Time.deltaTime * smooth *100); // dampen direction to straight ahead, no roll
//        Quaternion relativeRotation =  Quaternion.Inverse(newDirection) * Ship.rotation;


//        newDirection = Quaternion.Slerp(newDirection, Quaternion.Euler(0, Ship.rotation.y, Ship.rotation.z) , Time.deltaTime * smooth *100); // dampen direction to straight ahead, no roll


//        Vector3 angles = relativeRotation.eulerAngles;
//        Debug.Log("angle(" + angles.x + ", " + angles.y + ", " + angles.z + ")");
//        newDirection = Quaternion.Euler(angles.x, angles.y, -angles.z);
        newDirection = Ship.rotation * newDirection;
        Ship.rotation = (Quaternion.Slerp(Ship.rotation, newDirection,  Time.deltaTime * smooth));
//        Ship.rotation = (Quaternion.Slerp(Ship.rotation, Quaternion.Euler(0.0f, Ship.rotation.y,Ship.rotation.z),  Time.deltaTime * smooth));
        //Ship.Rotate( Input.GetAxis("Vertical"), 0.0f, -Input.GetAxis("Horizontal") );
        }

        if (mode == 1)  
        {
            speed = Mathf.Min(maxSpeed,speed+acceleration+(acceleration*speed));
            if (!isOff) {
                if  (speed < maxSpeed*.01)
                {
                    animator.SetTrigger("isHover");
                } else {
                    animator.SetTrigger("isAccelerating");
                    animator.ResetTrigger("isDecelerating");
                    animator.ResetTrigger("isHover");
                }
            }
        }
            else 
        {
            speed = Mathf.Max(0.0f,speed-deceleration-(deceleration*speed));
            if (!isOff) {
                if  (speed < maxSpeed*.01)
                {
                    animator.SetTrigger("isHover");
                } else {
                    animator.SetTrigger("isDecelerating");
                    animator.ResetTrigger("isAccelerating");
                    animator.ResetTrigger("isHover");
                }
            }
        }

        if  (speed > maxSpeed*.5)
        {
            animator.SetTrigger("isHighSpeed");
        }
        else
        {
            animator.ResetTrigger("isHighSpeed");
        }


        Vector3 VehicleVel = m_Rigidbody.velocity;
        vspeed = Mathf.Max(0.0f,vspeed-deceleration);
        float gravity = 9.81f * Time.deltaTime;
        LastFrameVel.y -= gravity; //add gravity
        //m_Rigidbody.AddForce(transform.up * 9.8f);
        Ship.position += transform.up * Time.deltaTime * vspeed;
        LastFrameVel = VehicleVel;
        if (dropreset==1) {
            if (Time.time - dropTime >1) {
                dropreset = 2;
            }
        }
        if (dropreset==2) {
            dropTime = 0;
            this.transform.localPosition = joystickSpawnPosition;
            this.transform.localRotation = joystickSpawnRotation;
            m_Rigidbody.ResetInertiaTensor();
            m_Rigidbody.velocity = new Vector3(0.0f,0.0f,0.0f);
            m_Rigidbody.angularVelocity = new Vector3(0.0f,0.0f,0.0f);
            dropreset=0;
        }
    }

    void OnPickupUseDown()
    {
        mode = 1;
    }

    void OnPickupUseUp()
    {
        mode = 0;
    }

    void OnPickup()
    {
        TakeOwnership();
        //player = Networking.LocalPlayer;
        //Networking.SetOwner(player, gameObject);
        vspeed = 2.0f;
        Ship.Rotate( Input.GetAxis("Vertical"), 0.0f, -Input.GetAxis("Horizontal") );
        dropreset = 0;
        animator.ResetTrigger("isOff");
        animator.SetTrigger("isHover");
        isOff=false;
    }

    void OnDrop()
    {
        dropreset = 1;
        dropTime = Time.time;
        // reset position of joystick to original
        this.speed = 0.0f;
        this.mode = 0;
        this.vspeed = 0.0f;
        m_Rigidbody.ResetInertiaTensor();
        m_Rigidbody.velocity = new Vector3(0.0f,0.0f,0.0f);
        m_Rigidbody.angularVelocity = new Vector3(0.0f,0.0f,0.0f);
        isOff = true;
        animator.SetTrigger("isOff");
        animator.ResetTrigger("isHover");
        animator.ResetTrigger("isAccelerating");
        animator.ResetTrigger("isDecelerating");
        animator.ResetTrigger("isHighSpeed");

    }

    public void TakeOwnership()
    {
        Debug.Log("Take Ship Ownership ");
        Debug.Log("From "+Networking.GetOwner(gameObject).displayName + " and assign it to "+Networking.LocalPlayer.displayName);
   //     if (Networking.IsMaster || !_masterOnly)
   //     {
            if (!Networking.IsOwner(gameObject))
            {
                Networking.SetOwner(Networking.LocalPlayer, gameObject);
            }
            if (!Networking.IsOwner(Ship.gameObject))
            {
                Networking.SetOwner(Networking.LocalPlayer, Ship.gameObject);
            }
//     }
    }

}
